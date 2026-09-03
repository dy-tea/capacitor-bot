use serenity::all::*;

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use capacitor::model::BuildProgress;
use capacitor::recipe::Recipe;

use crate::jobs::Trainer;
use crate::query::QueryEngine;
use crate::recipe_edit;
use crate::store::{ModelMeta, Store};
use crate::util::{clip, format_size};

pub struct Data {
    pub store: Arc<Mutex<Store>>,
    pub trainer: Trainer,
    pub query: QueryEngine,
    /// Maps the message id of a recipe reply to the file it is synced to, so
    /// that `message_update` events can keep the recipe file in step with edits
    /// to that message.
    recipe_sessions: recipe_edit::RecipeSessions,
}

impl Data {
    pub fn new(store: Arc<Mutex<Store>>, trainer: Trainer, query: QueryEngine) -> Self {
        Self {
            store,
            trainer,
            query,
            recipe_sessions: recipe_edit::new_sessions(),
        }
    }
}

pub struct Bot {
    pub data: Arc<Data>,
}

#[serenity::async_trait]
impl EventHandler for Bot {
    async fn ready(&self, ctx: Context, ready: Ready) {
        let guild_count = ready.guilds.len();

        sync_commands(&ctx, &ready).await;

        println!("capacitor-bot ready and serving {guild_count} guilds");
    }

    /// Keeps recipe files in sync when a user edits the message they replied
    /// with a recipe to a `/recipe create` or `/recipe edit` prompt.
    async fn message_update(
        &self,
        ctx: Context,
        _old: Option<Message>,
        _new: Option<Message>,
        event: MessageUpdateEvent,
    ) {
        recipe_edit::sync_edited_message(&ctx, &self.data.recipe_sessions, &event).await;
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Autocomplete(cmd) = interaction {
            let _ = self.autocomplete(&ctx, &cmd).await;
            return;
        }

        let Interaction::Command(cmd) = interaction else {
            return;
        };

        let result = match cmd.data.name.as_str() {
            "dataset" => self.dataset(&ctx, &cmd).await,
            "recipe" => self.recipe(&ctx, &cmd).await,
            "train" => self.train(&ctx, &cmd).await,
            "query" => self.query(&ctx, &cmd).await,
            "list" => self.list(&ctx, &cmd).await,
            "show" => self.show(&ctx, &cmd).await,
            "delete" => self.delete(&ctx, &cmd).await,
            "about" => self.about(&ctx, &cmd).await,
            name => {
                let _ = reply(&ctx, &cmd, format!("Unknown command `{name}`.")).await;
                Ok(())
            }
        };

        if let Err(err) = result {
            eprintln!("command `{}` failed: {err}", cmd.data.name);
            let _ = cmd
                .create_followup(
                    &ctx,
                    CreateInteractionResponseFollowup::new()
                        .content("Something went wrong while processing that command."),
                )
                .await;
        }
    }
}

impl Bot {
    async fn dataset(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        // `/dataset` is a command group; dispatch on its subcommand.
        let Some(sub) = cmd.data.options.first() else {
            reply(ctx, cmd, "Usage: `/dataset upload|list|delete`.").await?;
            return Ok(());
        };

        let CommandDataOptionValue::SubCommand(opts) = &sub.value else {
            reply(ctx, cmd, "Usage: `/dataset upload|list|delete`.").await?;
            return Ok(());
        };

        match sub.name.as_str() {
            "upload" => self.dataset_upload(ctx, cmd, opts).await,
            "list" => self.dataset_list(ctx, cmd).await,
            "delete" => self.dataset_delete(ctx, cmd, opts).await,
            other => {
                reply(ctx, cmd, format!("Unknown dataset subcommand `{other}`.")).await?;
                Ok(())
            }
        }
    }

    async fn dataset_upload(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(dataset) = cmd.data.resolved.attachments.values().next() else {
            reply(ctx, cmd, "Please attach a text file to upload.").await?;
            return Ok(());
        };

        let bytes = dataset.download().await?;

        // An explicit `name` overrides the attachment filename, letting users
        // pick a stable identifier and avoid `unique_path` suffixes.
        let name = option_str(options, "name")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| dataset.filename.clone());

        let saved = self
            .data
            .store
            .lock()
            .unwrap()
            .save_dataset(namespace, &name, &bytes)?;

        let file_name = saved
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("dataset")
            .to_string();

        reply(
            ctx,
            cmd,
            format!(
                "Uploaded dataset as `{file_name}` ({}). Use it with \
                 `/train dataset:{file_name}`.",
                format_size(bytes.len() as f64)
            ),
        )
        .await?;

        Ok(())
    }

    async fn dataset_list(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);
        let datasets = self.data.store.lock().unwrap().list_datasets(namespace);

        if datasets.is_empty() {
            reply(
                ctx,
                cmd,
                "No datasets uploaded in this server yet. Use `/dataset upload`.",
            )
            .await?;
            return Ok(());
        }

        let mut ns = datasets
            .iter()
            .map(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("dataset");
                let size = p.metadata().map(|m| m.len()).unwrap_or(0);
                (name, size)
            })
            .collect::<Vec<_>>();

        ns.sort_by(|a, b| a.0.cmp(b.0));

        let lines = ns
            .iter()
            .map(|p| format!("- `{}` ({})", p.0, format_size(p.1 as f64)))
            .collect::<Vec<_>>()
            .join("\n");

        reply(ctx, cmd, format!("Datasets in this server:\n{lines}")).await?;

        Ok(())
    }

    async fn dataset_delete(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(name) = option_str(options, "name") else {
            reply(ctx, cmd, "Missing required `name` option.").await?;
            return Ok(());
        };

        let removed = self
            .data
            .store
            .lock()
            .unwrap()
            .delete_dataset(namespace, &name)?;

        if removed {
            reply(ctx, cmd, format!("Deleted dataset `{name}`.")).await?;
        } else {
            reply(
                ctx,
                cmd,
                format!("No dataset named `{name}` in this server."),
            )
            .await?;
        }

        Ok(())
    }

    async fn recipe(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        // `/recipe` is a command group; dispatch on its subcommand.
        let Some(sub) = cmd.data.options.first() else {
            reply(ctx, cmd, "Usage: `/recipe upload|list|info`.").await?;
            return Ok(());
        };

        let CommandDataOptionValue::SubCommand(opts) = &sub.value else {
            reply(ctx, cmd, "Usage: `/recipe upload|list|info`.").await?;
            return Ok(());
        };

        match sub.name.as_str() {
            "create" => self.recipe_create(ctx, cmd, opts).await,
            "edit" => self.recipe_edit(ctx, cmd, opts).await,
            "upload" => self.recipe_upload(ctx, cmd, opts).await,
            "list" => self.recipe_list(ctx, cmd).await,
            "info" => self.recipe_info(ctx, cmd, opts).await,
            "delete" => self.recipe_delete(ctx, cmd, opts).await,
            other => {
                reply(ctx, cmd, format!("Unknown recipe subcommand `{other}`.")).await?;
                Ok(())
            }
        }
    }

    async fn recipe_upload(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(attachment) = cmd.data.resolved.attachments.values().next() else {
            reply(ctx, cmd, "Please attach a recipe to upload.").await?;
            return Ok(());
        };

        let bytes = attachment.download().await?;

        // An explicit `name` overrides the attachment filename, letting users
        // pick a stable identifier and avoid `unique_path` suffixes.
        let name = option_str(options, "name")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| attachment.filename.clone());

        let saved = self
            .data
            .store
            .lock()
            .unwrap()
            .save_capacitorfile(namespace, &name, &bytes)?;

        let file_name = saved
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("capacitorfile")
            .to_string();

        reply(
            ctx,
            cmd,
            format!(
                "Uploaded recipe as `{file_name}` ({}). Use it with \
                 `/train recipe:{file_name}`.",
                format_size(bytes.len() as f64)
            ),
        )
        .await?;

        Ok(())
    }

    async fn recipe_create(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let provided_name = option_str(options, "name").filter(|s| !s.is_empty());

        let prompt = "Reply to this message with the capacitorfile recipe content. \
            Editing your reply will update the recipe file."
            .to_string();

        let Some(reply) = recipe_edit::await_reply(ctx, cmd, prompt).await? else {
            cmd.channel_id
                .send_message(
                    ctx,
                    CreateMessage::new()
                        .content("Timed out waiting for your reply. No recipe was saved."),
                )
                .await?;
            return Ok(());
        };

        let content = recipe_edit::strip_fence(&reply.content);

        let name = match provided_name {
            Some(name) => name,
            None => match capacitor::recipe::Recipe::from_str(&content) {
                Ok(recipe) => recipe
                    .keys
                    .get("model.name")
                    .cloned()
                    .unwrap_or_else(|| String::from("recipe.capacitor")),
                Err(_) => String::from("recipe.capacitor"),
            },
        };

        let saved = self.data.store.lock().unwrap().save_capacitorfile(
            namespace,
            &name,
            content.as_bytes(),
        )?;

        recipe_edit::register_session(
            ctx,
            cmd,
            &self.data.recipe_sessions,
            &reply,
            saved,
            &content,
        )
        .await?;

        Ok(())
    }

    async fn recipe_edit(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(name) = option_str(options, "name") else {
            reply(ctx, cmd, "Missing required `name` option.").await?;
            return Ok(());
        };

        let path = {
            let store = self.data.store.lock().unwrap();
            store.find_capacitorfile(namespace, &name)
        };

        let Some(path) = path else {
            reply(
                ctx,
                cmd,
                format!("No recipe named `{name}` in this server."),
            )
            .await?;
            return Ok(());
        };

        let current = std::fs::read_to_string(&path)?;

        let prompt = format!(
            "**Current recipe `{name}`**\n\nReply to this message with the updated \
             capacitorfile recipe content. Editing your reply will update the \
             recipe file.\n\n```\n{content}\n```",
            content = clip(&current, 1800),
        );

        let Some(reply) = recipe_edit::await_reply(ctx, cmd, prompt).await? else {
            cmd.channel_id
                .send_message(
                    ctx,
                    CreateMessage::new()
                        .content("Timed out waiting for your reply. The recipe was not changed."),
                )
                .await?;
            return Ok(());
        };

        let content = recipe_edit::strip_fence(&reply.content);

        std::fs::write(&path, content.as_bytes())?;

        recipe_edit::register_session(ctx, cmd, &self.data.recipe_sessions, &reply, path, &content)
            .await?;

        Ok(())
    }

    async fn recipe_list(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);
        let files = self
            .data
            .store
            .lock()
            .unwrap()
            .list_capacitorfiles(namespace);

        if files.is_empty() {
            reply(
                ctx,
                cmd,
                "No recipes uploaded in this server yet. Use `/recipe upload`.",
            )
            .await?;
            return Ok(());
        }

        let mut ns = files
            .iter()
            .map(|p| {
                let name = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("capacitorfile");
                let size = p.metadata().map(|m| m.len()).unwrap_or(0);
                (name, size)
            })
            .collect::<Vec<_>>();

        ns.sort_by(|a, b| a.0.cmp(b.0));

        let lines = ns
            .iter()
            .map(|p| format!("- `{}` ({})", p.0, format_size(p.1 as f64)))
            .collect::<Vec<_>>()
            .join("\n");

        reply(ctx, cmd, format!("Recipe in this server:\n{lines}")).await?;

        Ok(())
    }

    async fn recipe_info(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(name) = option_str(options, "name") else {
            reply(ctx, cmd, "Missing required `name` option.").await?;
            return Ok(());
        };

        let path = {
            let store = self.data.store.lock().unwrap();
            store.find_capacitorfile(namespace, &name)
        };

        let Some(path) = path else {
            reply(
                ctx,
                cmd,
                format!("No recipe named `{name}` in this server."),
            )
            .await?;
            return Ok(());
        };

        let content = std::fs::read_to_string(&path)?;

        let size = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or_else(|_| content.len() as u64);

        // If it parses as a recipe, show which datasets it references and
        // whether each is uploaded; otherwise note it isn't a valid recipe.
        let references = match Recipe::from_str(&content) {
            Ok(recipe) => {
                let store = self.data.store.lock().unwrap();
                recipe
                    .files
                    .iter()
                    .filter_map(|f| f.path.file_name().and_then(|n| n.to_str()))
                    .map(|f| {
                        if store.find_dataset(namespace, f).is_some() {
                            format!("- `{f}` (uploaded)")
                        } else {
                            format!("- `{f}` (missing — `/dataset upload` it first)")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            Err(_) => String::from("- *(not a valid recipe; saved as-is)*"),
        };

        let body = format!(
            "**{name}** ({size} bytes)\n\n**References**\n{references}\n\n```\n{content}\n```",
            content = clip(&content, 1900),
        );

        reply(ctx, cmd, clip(&body, 2000)).await?;
        Ok(())
    }

    async fn recipe_delete(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(name) = option_str(options, "name") else {
            reply(ctx, cmd, "Missing required `name` option.").await?;
            return Ok(());
        };

        let removed = self
            .data
            .store
            .lock()
            .unwrap()
            .delete_capacitorfile(namespace, &name)?;

        if removed {
            reply(ctx, cmd, format!("Deleted recipe `{name}`.")).await?;
        } else {
            reply(
                ctx,
                cmd,
                format!("No recipe named `{name}` in this server."),
            )
            .await?;
        }

        Ok(())
    }

    async fn train(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        // model name is optional; if not provided, derive from recipe/dataset name
        let model_name = match option_str(&cmd.data.options, "model") {
            Some(name) if !name.is_empty() => name,
            _ => {
                // Derive from recipe or dataset name
                let source_name = option_str(&cmd.data.options, "recipe");
                match source_name {
                    Some(name) => name,
                    None => {
                        reply(
                            ctx,
                            cmd,
                            "Missing `model` name and no `recipe` provided to derive it from.",
                        )
                        .await?;
                        return Ok(());
                    }
                }
            }
        };

        let source = option_str(&cmd.data.options, "recipe");

        let Some(source) = source else {
            reply(
                ctx,
                cmd,
                "Missing input. Provide either `dataset` (a raw corpus to train on \
                 directly) or `recipe` (a recipe referencing uploaded datasets).",
            )
            .await?;
            return Ok(());
        };

        cmd.defer(ctx).await?;

        let mut token_expired = false;

        let (recipe, document_count, dataset_paths) =
            match build_recipe(&self.data, namespace, &model_name, &cmd.data.options) {
                Ok(Some(config)) => config,
                Ok(None) => {
                    deliver_update(
                        ctx,
                        cmd,
                        &mut token_expired,
                        &format!(
                            "No usable recipe `{source}` (unknown, empty, or \
                         split into zero documents)."
                        ),
                    )
                    .await?;
                    return Ok(());
                }
                Err(err) => {
                    deliver_update(
                        ctx,
                        cmd,
                        &mut token_expired,
                        &format!("Could not build training config: {err}"),
                    )
                    .await?;
                    return Ok(());
                }
            };

        let recipe_text = recipe.to_string();

        let total_experts = recipe.experts.num_total;
        let active_experts = recipe.experts.num_active;
        let centroids = recipe.experts.num_centroids;

        let models_dir = self.data.store.lock().unwrap().models_dir(namespace);
        std::fs::create_dir_all(&models_dir)?;

        let output_path = models_dir.join(format!("{model_name}.capacitor"));

        let mut job = match self
            .data
            .trainer
            .submit(namespace, recipe, output_path.clone())
            .await
        {
            Ok(job) => job,
            Err(err) => {
                deliver_update(
                    ctx,
                    cmd,
                    &mut token_expired,
                    &format!("Could not start training: {err}"),
                )
                .await?;
                return Ok(());
            }
        };

        let mut message = format!(
            "Training `{model_name}` (job `#{}`) on {document_count} documents: \
             {total_experts} experts / {active_experts} active / {centroids} centroids...",
            job.id
        );
        deliver_update(ctx, cmd, &mut token_expired, &message).await?;

        let mut last_phase: &str = "";
        let mut last_update = std::time::Instant::now();
        let min_interval = std::time::Duration::from_secs(2);

        loop {
            if let Ok(event) = job.progress.try_recv() {
                let phase = progress_phase_label(&event);
                let phase_changed = phase != last_phase;
                last_phase = phase;

                let fresh = last_update.elapsed() >= min_interval;
                if phase_changed || (fresh && !token_expired) {
                    message = format!(
                        "Training `{model_name}` (job `#{}`): {}",
                        job.id,
                        progress_phase(&event),
                    );
                    deliver_update(ctx, cmd, &mut token_expired, &message).await?;
                    last_update = std::time::Instant::now();
                }
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }

            if job.progress.is_closed() {
                break;
            }
        }

        let outcome = match job.result.await {
            Ok(Ok(_)) => {
                let meta = ModelMeta {
                    name: model_name.clone(),
                    path: output_path,
                    datasets: dataset_paths,
                    owner: namespace,
                    created_at: std::time::UNIX_EPOCH
                        .elapsed()
                        .unwrap_or_default()
                        .as_secs(),
                    recipe: recipe_text,
                };

                self.data
                    .store
                    .lock()
                    .unwrap()
                    .register_model(namespace, meta)?;

                format!(
                    "Model `{model_name}` trained and saved. Query it with \
                     `/query model:{model_name} prompt:<your prompt>`."
                )
            }
            Ok(Err(err)) => format!("Training `{model_name}` failed: {err}"),
            Err(_) => format!("Training `{model_name}` was aborted before completing."),
        };

        deliver_update(ctx, cmd, &mut token_expired, &outcome).await?;

        Ok(())
    }

    async fn query(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(prompt) = option_str(&cmd.data.options, "prompt") else {
            reply(ctx, cmd, "Missing required `prompt` option.").await?;
            return Ok(());
        };

        let model_name = {
            let mut store = self.data.store.lock().unwrap();

            match option_str(&cmd.data.options, "model") {
                Some(name) => {
                    store.set_last_used(namespace, &name)?;
                    Some(name)
                }
                None => store.last_used(namespace),
            }
        };

        let Some(model_name) = model_name else {
            reply(ctx, cmd, "No `model` given and no model has been used in this server yet. Run `/query model:<name> ...` (autocomplete available) or `/list`.").await?;
            return Ok(());
        };

        let meta = {
            let store = self.data.store.lock().unwrap();
            store.get(namespace, &model_name)
        };

        let Some(meta) = meta else {
            reply(
                ctx,
                cmd,
                format!("No model named `{model_name}` in this server. Use `/list`."),
            )
            .await?;
            return Ok(());
        };

        cmd.defer(ctx).await?;

        let text = self
            .data
            .query
            .query(namespace, meta, prompt.clone())
            .await?;

        let response = format!("{prompt} {}", clip(&text, 1900));

        cmd.edit_response(
            ctx,
            EditInteractionResponse::new().content(clip(&response, 2000)),
        )
        .await?;

        Ok(())
    }

    async fn list(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);
        let mut models = self.data.store.lock().unwrap().list(namespace);

        if models.is_empty() {
            reply(
                ctx,
                cmd,
                "No models in this server yet. Use `/dataset upload` and `/train` one.",
            )
            .await?;
            return Ok(());
        }

        models.sort_by(|a, b| a.name.cmp(&b.name));

        let lines = models
            .iter()
            .map(|m| {
                format!(
                    "- `{}` (owner <@{}>, {} datasets)",
                    m.name,
                    m.owner,
                    m.datasets.len()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        reply(ctx, cmd, format!("Models in this server:\n{lines}")).await?;

        Ok(())
    }

    async fn autocomplete(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        // `autocomplete()` recurses through subcommands and returns the leaf
        // option that is currently focused, so we know exactly which selector
        // to populate. A dataset selector never mixes capacitorfile recipes,
        // and a model selector never shows either.
        let Some(auto) = cmd.data.autocomplete() else {
            return Ok(());
        };

        let option_name = auto.name;
        let query = auto.value.to_lowercase();

        let names: Vec<String> = {
            let store = self.data.store.lock().unwrap();

            match (cmd.data.name.as_str(), option_name) {
                ("train", "recipe") | ("recipe", "name") => store
                    .list_capacitorfiles(namespace)
                    .into_iter()
                    .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                    .collect(),
                ("dataset", "name") => store
                    .list_datasets(namespace)
                    .into_iter()
                    .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                    .collect(),
                _ => store.list(namespace).into_iter().map(|m| m.name).collect(),
            }
        };

        let choices = names
            .into_iter()
            .filter(|name| name.to_lowercase().contains(&query))
            .map(|name| AutocompleteChoice::new(name.clone(), name))
            .take(25)
            .collect::<Vec<_>>();

        cmd.create_response(
            ctx,
            CreateInteractionResponse::Autocomplete(
                CreateAutocompleteResponse::new().set_choices(choices),
            ),
        )
        .await
        .map_err(anyhow::Error::from)?;

        Ok(())
    }

    async fn show(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(model_name) = option_str(&cmd.data.options, "model") else {
            reply(ctx, cmd, "Missing required `model` option.").await?;
            return Ok(());
        };

        let meta = self.data.store.lock().unwrap().get(namespace, &model_name);

        let Some(meta) = meta else {
            reply(
                ctx,
                cmd,
                format!("No model named `{model_name}` in this server."),
            )
            .await?;
            return Ok(());
        };

        reply(
            ctx,
            cmd,
            format!(
                "**{name}**\n- owner: <@{owner}>\n- datasets: {datasets}\n\n```\n{recipe}\n```",
                name = meta.name,
                owner = meta.owner,
                datasets = meta
                    .datasets
                    .iter()
                    .map(|d| d.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                recipe = meta.recipe
            ),
        )
        .await?;

        Ok(())
    }

    async fn delete(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(model_name) = option_str(&cmd.data.options, "model") else {
            reply(ctx, cmd, "Missing required `model` option.").await?;
            return Ok(());
        };

        let removed = self
            .data
            .store
            .lock()
            .unwrap()
            .delete(namespace, &model_name)?;

        if removed {
            reply(ctx, cmd, format!("Deleted model `{model_name}`.")).await?;
        } else {
            reply(
                ctx,
                cmd,
                format!("No model named `{model_name}` in this server."),
            )
            .await?;
        }

        Ok(())
    }

    async fn about(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        reply(
            ctx,
            cmd,
            "**Capacitor bot**\n\
             Commands: `/dataset` (`upload`/`list`/`delete`), `/recipe` (`create`/`edit`/`upload`/`list`/`info`/`delete`), \
             `/train`, `/query`, `/list`, `/show`, `/delete`.",
        )
        .await?;

        Ok(())
    }
}

/// Returns the storage namespace for an interaction.
fn namespace(cmd: &CommandInteraction) -> u64 {
    cmd.guild_id
        .map(|g| g.get())
        .unwrap_or_else(|| cmd.user.id.get())
}

fn option_str(options: &[CommandDataOption], name: &str) -> Option<String> {
    options
        .iter()
        .find(|o| o.name == name)
        .and_then(|o| match &o.value {
            CommandDataOptionValue::String(s) => Some(s.clone()),
            _ => None,
        })
}

fn option_int(options: &[CommandDataOption], name: &str) -> Option<i64> {
    options
        .iter()
        .find(|o| o.name == name)
        .and_then(|o| match &o.value {
            CommandDataOptionValue::Integer(v) => Some(*v),
            _ => None,
        })
}

async fn reply(
    ctx: &Context,
    cmd: &CommandInteraction,
    text: impl Into<String>,
) -> anyhow::Result<()> {
    cmd.create_response(
        ctx,
        CreateInteractionResponse::Message(CreateInteractionResponseMessage::new().content(text)),
    )
    .await
    .map_err(anyhow::Error::from)?;

    Ok(())
}

fn commands() -> Vec<CreateCommand> {
    let commands = vec![
        CreateCommand::new("dataset")
            .description("Manage text datasets for this server")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "upload",
                    "Upload a text dataset (raw corpus) for training",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::Attachment,
                        "file",
                        "The text file to use as training data",
                    )
                    .required(true),
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::String,
                    "name",
                    "Custom name for the dataset (avoids collisions)",
                )),
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::SubCommand,
                "list",
                "List datasets in this server",
            ))
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "delete",
                    "Delete a dataset from this server",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "name",
                        "Name of the dataset to delete",
                    )
                    .required(true)
                    .set_autocomplete(true),
                ),
            ),
        CreateCommand::new("recipe")
            .description("Manage capacitorfile recipes for this server")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "create",
                    "Create a capacitorfile recipe by replying with its content",
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::String,
                    "name",
                    "Custom name for the capacitorfile (avoids collisions)",
                )),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "edit",
                    "Edit an existing capacitorfile recipe by replying",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "name",
                        "Name of the capacitorfile to edit",
                    )
                    .required(true)
                    .set_autocomplete(true),
                ),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "upload",
                    "Upload a capacitorfile recipe",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::Attachment,
                        "file",
                        "The capacitorfile recipe to upload",
                    )
                    .required(true),
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::String,
                    "name",
                    "Custom name for the capacitorfile (avoids collisions)",
                )),
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::SubCommand,
                "list",
                "List capacitorfile recipes in this server",
            ))
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "info",
                    "Show details of a capacitorfile recipe",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "name",
                        "Name of the capacitorfile",
                    )
                    .required(true)
                    .set_autocomplete(true),
                ),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "delete",
                    "Delete a capacitorfile recipe from this server",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "name",
                        "Name of the capacitorfile to delete",
                    )
                    .required(true)
                    .set_autocomplete(true),
                ),
            ),
        CreateCommand::new("train")
            .description("Train a model from a corpus or a capacitorfile recipe")
            .add_option(CreateCommandOption::new(
                CommandOptionType::String,
                "model",
                "Name for the new model (defaults to name set by recipe)",
            ))
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "recipe",
                    "Name of an uploaded capacitorfile recipe to train from",
                )
                .set_autocomplete(true),
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::String,
                "split",
                "Optional document delimiter",
            )),
        CreateCommand::new("query")
            .description("Query a trained model for text generation (last-used model if omitted)")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "prompt",
                    "The prompt to generate from",
                )
                .required(true),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "model",
                    "Name of the model (defaults to last used)",
                )
                .set_autocomplete(true),
            ),
        CreateCommand::new("list").description("List models available in this server"),
        CreateCommand::new("show")
            .description("Show details of a model")
            .add_option(
                CreateCommandOption::new(CommandOptionType::String, "model", "Name of the model")
                    .required(true)
                    .set_autocomplete(true),
            ),
        CreateCommand::new("delete")
            .description("Delete a model")
            .add_option(
                CreateCommandOption::new(CommandOptionType::String, "model", "Name of the model")
                    .required(true)
                    .set_autocomplete(true),
            ),
        CreateCommand::new("about").description("About this bot"),
    ];

    commands.into_iter().map(user_app_command).collect()
}

fn user_app_command(cmd: CreateCommand) -> CreateCommand {
    cmd.add_context(InteractionContext::Guild)
        .add_context(InteractionContext::BotDm)
        .add_context(InteractionContext::PrivateChannel)
        .add_integration_type(InstallationContext::Guild)
        .add_integration_type(InstallationContext::User)
}

/// Re-register the bot's command set on every launch so renames, removals and
/// additions all take effect from a clean slate.
async fn sync_commands(ctx: &Context, ready: &Ready) {
    let commands = commands();
    let app_id = ready.application.id;

    if let Ok(existing) = ctx.http.get_global_commands().await {
        for command in existing.iter().filter(|c| c.application_id == app_id) {
            if let Err(err) = ctx.http.delete_global_command(command.id).await {
                eprintln!("failed to delete global command {}: {err}", command.id);
            }
        }
    }

    for guild in &ready.guilds {
        if let Ok(existing) = ctx.http.get_guild_commands(guild.id).await {
            for command in existing.iter().filter(|c| c.application_id == app_id) {
                if let Err(err) = ctx.http.delete_guild_command(guild.id, command.id).await {
                    eprintln!(
                        "failed to delete guild command {} in guild {}: {err}",
                        command.id, guild.id,
                    );
                }
            }
        }
    }

    if let Err(err) = ctx.http.create_global_commands(&commands).await {
        eprintln!("failed to register global commands: {err}");
    }
}

/// Report a training status line. While the interaction token is still valid
/// (the first ~15 minutes) this edits the deferred response; once that stops
/// working it posts a fresh channel message instead, which is independent of the
/// interaction token and so survives trainings that outlive the token lifetime.
async fn deliver_update(
    ctx: &Context,
    cmd: &CommandInteraction,
    token_expired: &mut bool,
    msg: &str,
) -> anyhow::Result<()> {
    if !*token_expired {
        if cmd
            .edit_response(ctx, EditInteractionResponse::new().content(msg))
            .await
            .is_ok()
        {
            return Ok(());
        }

        *token_expired = true;
    }

    cmd.channel_id
        .send_message(ctx, CreateMessage::new().content(msg))
        .await?;

    Ok(())
}

fn build_recipe(
    data: &Data,
    namespace: u64,
    model_name: &str,
    options: &[CommandDataOption],
) -> anyhow::Result<Option<(Recipe, usize, Vec<PathBuf>)>> {
    let path = {
        let store = data.store.lock().unwrap();
        match store.find_capacitorfile(namespace, model_name) {
            Some(path) => path,
            None => return Ok(None),
        }
    };

    let raw = std::fs::read_to_string(&path)?;

    let recipe = Recipe::from_str(&raw).map_err(|err| {
        anyhow::anyhow!("`{model_name}` did not parse as a capacitorfile recipe: {err}")
    })?;

    resolve_recipe(data, namespace, model_name, recipe, options)
}

/// Build a training configuration from a capacitorfile recipe, resolving its
/// dataset references to datasets uploaded in the current server.
fn resolve_recipe(
    data: &Data,
    namespace: u64,
    model_name: &str,
    mut recipe: Recipe,
    options: &[CommandDataOption],
) -> anyhow::Result<Option<(Recipe, usize, Vec<PathBuf>)>> {
    let mut total_documents = 0usize;
    let mut dataset_paths: Vec<PathBuf> = Vec::new();

    for file in &mut recipe.files {
        let Some(file_name) = file.path.file_name().and_then(|n| n.to_str()) else {
            anyhow::bail!(
                "recipe references a dataset with an unreadable name: {}",
                file.path.display()
            );
        };

        let resolved = {
            let store = data.store.lock().unwrap();
            match store.find_dataset(namespace, file_name) {
                Some(path) => path,
                None => anyhow::bail!(
                    "recipe references `{file_name}`, but no dataset with that name is uploaded in this server. `/dataset upload` it first."
                ),
            }
        };

        file.path = resolved.clone();
        total_documents += count_documents(&file.path, &file.delimiter)?;
        dataset_paths.push(resolved);
    }

    if total_documents == 0 {
        return Ok(None);
    }

    if let Some(v) = option_int(options, "total_experts") {
        recipe.experts.num_total = v as usize;
    }

    if let Some(v) = option_int(options, "active_experts") {
        recipe.experts.num_active = v as usize;
    }

    if let Some(v) = option_int(options, "centroids") {
        recipe.experts.num_centroids = v as usize;
    }

    clamp_clustering(&mut recipe, total_documents);

    recipe
        .keys
        .insert(String::from("model.name"), model_name.to_string());

    Ok(Some((recipe, total_documents, dataset_paths)))
}

/// Clamp `total_experts`, `active_experts` and `centroids` so that
/// `total_experts * centroids` never exceeds the number of documents.
/// Values of `0` are left as-is, since `clusterize` treats them as "auto".
fn clamp_clustering(recipe: &mut Recipe, documents: usize) {
    if recipe.experts.num_total > 0 {
        recipe.experts.num_total = recipe.experts.num_total.min(documents).max(1);
    }

    if recipe.experts.num_active > 0 {
        recipe.experts.num_active = recipe
            .experts
            .num_active
            .min(recipe.experts.num_total.max(1))
            .max(1);
    }

    let per_expert_cap = match recipe.experts.num_total {
        0 => documents,
        e => documents / e,
    }
    .max(1);

    if recipe.experts.num_centroids > 0 {
        recipe.experts.num_centroids = recipe.experts.num_centroids.min(per_expert_cap).max(1);
    }
}

/// Count how many non-empty documents a dataset contains when split by the
/// given delimiter, mirroring `Recipe::build`.
fn count_documents(path: &std::path::Path, delimiter: &str) -> anyhow::Result<usize> {
    let raw = std::fs::read_to_string(path)?;

    Ok(if delimiter.is_empty() {
        1
    } else {
        raw.split(delimiter)
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .count()
    })
}

/// Render a capacitor build progress event as a human-readable phase, e.g.
/// `"experts 3/4 (75%)"` or `"building tokens map"`.
fn progress_phase(event: &BuildProgress) -> String {
    let pct = |current: u64, total: u64| -> String {
        if total == 0 {
            String::new()
        } else {
            format!(" ({:.0}%)", current as f64 / total as f64 * 100.0)
        }
    };

    match event {
        BuildProgress::ReadFiles { current, total } => {
            format!(
                "reading files: {current}/{total}{}",
                pct(*current as u64, *total as u64)
            )
        }
        BuildProgress::PreTokenize { current, total } => {
            format!(
                "pre-tokenizing: {current}/{total} bytes{}",
                pct(*current, *total)
            )
        }
        BuildProgress::FitTokenizer { current, total } => {
            format!(
                "fitting tokenizer: {current}/{total} tokens{}",
                pct(*current as u64, *total as u64)
            )
        }
        BuildProgress::BuildTokensMap => "building tokens map".to_string(),
        BuildProgress::BuildSharedTransitions => "building shared transitions".to_string(),
        BuildProgress::ClusterizeDatasets => "clustering documents".to_string(),
        BuildProgress::BuildExperts { current, total } => {
            format!(
                "experts: {current}/{total}{}",
                pct(*current as u64, *total as u64)
            )
        }
        BuildProgress::Done => "finalizing".to_string(),
    }
}

/// Stable, copyable label for a build phase. Used to detect phase transitions so
/// progress updates aren't posted on every tick of a high-frequency phase.
fn progress_phase_label(event: &BuildProgress) -> &'static str {
    match event {
        BuildProgress::ReadFiles { .. } => "read_files",
        BuildProgress::PreTokenize { .. } => "pre_tokenize",
        BuildProgress::FitTokenizer { .. } => "fit_tokenizer",
        BuildProgress::BuildTokensMap => "tokens_map",
        BuildProgress::BuildSharedTransitions => "shared_transitions",
        BuildProgress::ClusterizeDatasets => "clusterize",
        BuildProgress::BuildExperts { .. } => "experts",
        BuildProgress::Done => "done",
    }
}
