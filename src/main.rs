use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use serenity::all::*;

use capacitor::model::BuildProgress;
use capacitor::recipe::{Experts, File, Recipe, Tokenizer};

use crate::jobs::Trainer;
use crate::query::QueryEngine;
use crate::store::{ModelMeta, Store};

mod jobs;
mod query;
mod store;

/// What to train on: either a raw uploaded dataset (treated as a corpus) or a
/// capacitorfile recipe (which references other uploaded datasets).
enum TrainSource {
    Dataset(String),
    Capacitorfile(String),
}

impl TrainSource {
    fn name(&self) -> &str {
        match self {
            TrainSource::Dataset(n) => n,
            TrainSource::Capacitorfile(n) => n,
        }
    }
}

struct Data {
    store: Arc<Mutex<Store>>,
    trainer: Trainer,
    query: QueryEngine,
}

struct Bot {
    data: Arc<Data>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let token =
        std::env::var("DISCORD_TOKEN").expect("DISCORD_TOKEN environment variable is required");

    let store_path =
        PathBuf::from(std::env::var("CAPACITOR_DIR").unwrap_or_else(|_| String::from("data")));

    let train_workers = parse_env::<usize>("TRAIN_WORKERS", 2)?;
    let query_capacity = parse_env::<usize>("QUERY_CAPACITY", 2)?;
    let cache_size = parse_env::<usize>("CACHE_SIZE", 16)?;

    let store = Arc::new(Mutex::new(Store::new(store_path)?));

    let data = Arc::new(Data {
        store,
        trainer: Trainer::spawn(train_workers),
        query: QueryEngine::new(query_capacity, cache_size),
    });

    let mut client = Client::builder(&token, GatewayIntents::non_privileged())
        .event_handler(Bot {
            data: Arc::clone(&data),
        })
        .await
        .map_err(anyhow::Error::from)?;

    client.start().await?;

    Ok(())
}

#[serenity::async_trait]
impl EventHandler for Bot {
    async fn ready(&self, ctx: Context, ready: Ready) {
        let commands = commands();
        let guild_count = ready.guilds.len();

        for guild in &ready.guilds {
            if let Err(err) = ctx.http.create_guild_commands(guild.id, &commands).await {
                eprintln!("failed to register commands for guild {}: {err}", guild.id);
            }
        }

        println!("capacitor-bot ready and serving {guild_count} guilds");
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
            "capacitorfile" => self.capacitorfile(&ctx, &cmd).await,
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
            reply(ctx, cmd, "Usage: `/dataset upload|list`.").await?;
            return Ok(());
        };

        let CommandDataOptionValue::SubCommand(opts) = &sub.value else {
            reply(ctx, cmd, "Usage: `/dataset upload|list`.").await?;
            return Ok(());
        };

        match sub.name.as_str() {
            "upload" => self.dataset_upload(ctx, cmd, opts).await,
            "list" => self.dataset_list(ctx, cmd).await,
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
                "Uploaded dataset as `{file_name}` ({} bytes). Use it with \
                 `/train dataset:{file_name}`.",
                bytes.len()
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

        let lines = datasets
            .iter()
            .map(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("dataset");
                format!(
                    "- `{name}` ({} bytes)",
                    p.metadata().map(|m| m.len()).unwrap_or(0)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        reply(ctx, cmd, format!("Datasets in this server:\n{lines}")).await?;

        Ok(())
    }

    async fn capacitorfile(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        // `/capacitorfile` is a command group; dispatch on its subcommand.
        let Some(sub) = cmd.data.options.first() else {
            reply(ctx, cmd, "Usage: `/capacitorfile upload|list|info`.").await?;
            return Ok(());
        };

        let CommandDataOptionValue::SubCommand(opts) = &sub.value else {
            reply(ctx, cmd, "Usage: `/capacitorfile upload|list|info`.").await?;
            return Ok(());
        };

        match sub.name.as_str() {
            "upload" => self.capacitorfile_upload(ctx, cmd, opts).await,
            "list" => self.capacitorfile_list(ctx, cmd).await,
            "info" => self.capacitorfile_info(ctx, cmd, opts).await,
            other => {
                reply(
                    ctx,
                    cmd,
                    format!("Unknown capacitorfile subcommand `{other}`."),
                )
                .await?;
                Ok(())
            }
        }
    }

    async fn capacitorfile_upload(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        options: &[CommandDataOption],
    ) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(attachment) = cmd.data.resolved.attachments.values().next() else {
            reply(ctx, cmd, "Please attach a capacitorfile recipe to upload.").await?;
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
                "Uploaded capacitorfile as `{file_name}` ({} bytes). Use it with \
                 `/train capacitorfile:{file_name}`.",
                bytes.len()
            ),
        )
        .await?;

        Ok(())
    }

    async fn capacitorfile_list(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
    ) -> anyhow::Result<()> {
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
                "No capacitorfiles uploaded in this server yet. Use `/capacitorfile upload`.",
            )
            .await?;
            return Ok(());
        }

        let lines = files
            .iter()
            .map(|p| {
                let name = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("capacitorfile");
                format!(
                    "- `{name}` ({} bytes)",
                    p.metadata().map(|m| m.len()).unwrap_or(0)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        reply(ctx, cmd, format!("Capacitorfiles in this server:\n{lines}")).await?;

        Ok(())
    }

    async fn capacitorfile_info(
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
                format!("No capacitorfile named `{name}` in this server."),
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

    async fn train(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(model_name) = option_str(&cmd.data.options, "model") else {
            reply(ctx, cmd, "Missing required `model` option.").await?;
            return Ok(());
        };

        // Exactly one of `dataset` (raw corpus) or `capacitorfile` (recipe) may
        // be supplied. They are mutually exclusive so the train selector never
        // mixes the two concepts.
        let source = option_str(&cmd.data.options, "capacitorfile")
            .map(TrainSource::Capacitorfile)
            .or_else(|| option_str(&cmd.data.options, "dataset").map(TrainSource::Dataset));

        let Some(source) = source else {
            reply(
                ctx,
                cmd,
                "Missing input. Provide either `dataset` (a raw corpus to train on \
                 directly) or `capacitorfile` (a recipe referencing uploaded datasets).",
            )
            .await?;
            return Ok(());
        };

        // Capture identifying info before `source` is moved into `build_recipe`.
        let source_kind = match &source {
            TrainSource::Dataset(_) => "dataset",
            TrainSource::Capacitorfile(_) => "capacitorfile",
        };
        let source_name = source.name().to_string();

        let (recipe, document_count, dataset_paths) = match build_recipe(
            &self.data,
            namespace,
            &model_name,
            source,
            &cmd.data.options,
        ) {
            Ok(Some(config)) => config,
            Ok(None) => {
                reply(
                    ctx,
                    cmd,
                    format!(
                        "No usable {source_kind} `{source_name}` (unknown, empty, or \
                         split into zero documents).",
                    ),
                )
                .await?;
                return Ok(());
            }
            Err(err) => {
                reply(ctx, cmd, format!("Could not build training config: {err}")).await?;
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

        let mut job = self
            .data
            .trainer
            .submit(namespace, recipe, output_path.clone())
            .await?;

        cmd.defer(ctx).await?;

        let mut message = format!(
            "Training `{model_name}` (job `#{}`) on {document_count} documents: {total_experts} experts / {active_experts} active / {centroids} centroids...",
            job.id
        );
        cmd.edit_response(ctx, EditInteractionResponse::new().content(&message))
            .await?;

        loop {
            if let Ok(event) = job.progress.try_recv() {
                message = format!(
                    "Training `{model_name}` (job `#{}`): {}",
                    job.id,
                    progress_phase(&event),
                );
                cmd.edit_response(ctx, EditInteractionResponse::new().content(&message))
                    .await?;
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }

            if job.progress.is_closed() {
                break;
            }
        }

        match job.result.await? {
            Ok(_) => {
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

                cmd.edit_response(ctx, EditInteractionResponse::new().content(format!(
                    "Model `{model_name}` trained and saved. Query it with `/query model:{model_name} prompt:<your prompt>`."
                ))).await?;
            }

            Err(err) => {
                cmd.edit_response(
                    ctx,
                    EditInteractionResponse::new()
                        .content(format!("Training `{model_name}` failed: {err}")),
                )
                .await?;
            }
        }

        Ok(())
    }

    async fn query(&self, ctx: &Context, cmd: &CommandInteraction) -> anyhow::Result<()> {
        let namespace = namespace(cmd);

        let Some(prompt) = option_str(&cmd.data.options, "prompt") else {
            reply(ctx, cmd, "Missing required `prompt` option.").await?;
            return Ok(());
        };

        let _seed = option_int(&cmd.data.options, "seed").map(|v| v as u64);

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
            .query(namespace, meta, prompt.clone(), _seed)
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
        let models = self.data.store.lock().unwrap().list(namespace);

        if models.is_empty() {
            reply(
                ctx,
                cmd,
                "No models in this server yet. Use `/dataset upload` and `/train` one.",
            )
            .await?;
            return Ok(());
        }

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
                ("train", "dataset") | ("capacitorfile", "name") => store
                    .list_capacitorfiles(namespace)
                    .into_iter()
                    .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                    .collect(),
                ("train", "capacitorfile") => store
                    .list_capacitorfiles(namespace)
                    .into_iter()
                    .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                    .collect(),
                // Any other autocomplete (query/show/delete `model`) is a model
                // name selector.
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
             Commands: `/dataset` (`upload`/`list`), `/capacitorfile` (`upload`/`list`/`info`), \
             `/train`, `/query`, `/list`, `/show`, `/delete`.",
        )
        .await?;

        Ok(())
    }
}

fn namespace(cmd: &CommandInteraction) -> u64 {
    cmd.guild_id
        .map(|g| g.get())
        .unwrap_or_else(|| cmd.user.id.get())
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

fn clip(text: &str, max: usize) -> String {
    let mut s: String = text.chars().take(max).collect();

    if text.len() > s.len() {
        s.push_str("\n... (truncated)");
    }

    s
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> anyhow::Result<T> {
    Ok(match std::env::var(name) {
        Ok(raw) => raw
            .parse::<T>()
            .map_err(|_| anyhow::anyhow!("invalid value for {name}: {raw}"))?,
        Err(_) => default,
    })
}

fn build_recipe(
    data: &Data,
    namespace: u64,
    model_name: &str,
    source: TrainSource,
    options: &[CommandDataOption],
) -> anyhow::Result<Option<(Recipe, usize, Vec<PathBuf>)>> {
    // What the user selected determines how the file is interpreted: a
    // capacitorfile is always a recipe (its `File`/`Split` lines reference
    // other uploaded datasets), while a dataset is always a raw corpus. This is
    // chosen by *where* the file lives, not by sniffing its contents, so the
    // train selector is unambiguous.
    match source {
        TrainSource::Capacitorfile(name) => {
            let path = {
                let store = data.store.lock().unwrap();
                match store.find_capacitorfile(namespace, &name) {
                    Some(path) => path,
                    None => return Ok(None),
                }
            };

            let raw = std::fs::read_to_string(&path)?;

            let recipe = Recipe::from_str(&raw).map_err(|err| {
                anyhow::anyhow!("`{name}` did not parse as a capacitorfile recipe: {err}")
            })?;

            resolve_recipe(data, namespace, model_name, recipe, options)
        }

        TrainSource::Dataset(name) => {
            let path = {
                let store = data.store.lock().unwrap();
                match store.find_dataset(namespace, &name) {
                    Some(path) => path,
                    None => return Ok(None),
                }
            };

            build_from_corpus(model_name, &path, options)
        }
    }
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

    if let Some(v) = option_int(options, "top_k") {
        recipe
            .keys
            .insert(String::from("model.inference.top_k"), v.to_string());
    }

    if let Some(v) = option_int(options, "max_tokens") {
        recipe
            .keys
            .insert(String::from("model.inference.max_tokens"), v.to_string());
    }

    recipe
        .keys
        .insert(String::from("model.name"), model_name.to_string());

    Ok(Some((recipe, total_documents, dataset_paths)))
}

/// Build a training configuration from a raw corpus dataset (no recipe).
fn build_from_corpus(
    model_name: &str,
    dataset_path: &std::path::Path,
    options: &[CommandDataOption],
) -> anyhow::Result<Option<(Recipe, usize, Vec<PathBuf>)>> {
    let delimiter = option_str(options, "split").unwrap_or_else(|| String::from("<|document|>"));

    let document_count = count_documents(dataset_path, &delimiter)?;

    if document_count == 0 {
        return Ok(None);
    }

    // Clamp clustering parameters so `total_experts * centroids` never exceeds
    // the number of documents in the dataset (see `clustering::clusterize`).
    let mut recipe = Recipe {
        tokenizer: Tokenizer {
            make_lowercase: true,
            force_alphanumeric: false,
            ..Default::default()
        },
        files: vec![File {
            path: dataset_path.to_path_buf(),
            delimiter,
            shuffle: false,
        }],
        experts: Experts {
            num_total: option_int(options, "total_experts").unwrap_or(4) as usize,
            num_active: option_int(options, "active_experts").unwrap_or(1) as usize,
            num_centroids: option_int(options, "centroids").unwrap_or(2) as usize,
        },
        ..Default::default()
    };

    recipe
        .keys
        .insert(String::from("model.name"), model_name.to_string());

    clamp_clustering(&mut recipe, document_count);

    if let Some(v) = option_int(options, "top_k") {
        recipe
            .keys
            .insert(String::from("model.inference.top_k"), v.to_string());
    }

    if let Some(v) = option_int(options, "max_tokens") {
        recipe
            .keys
            .insert(String::from("model.inference.max_tokens"), v.to_string());
    }

    Ok(Some((
        recipe,
        document_count,
        vec![dataset_path.to_path_buf()],
    )))
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

fn commands() -> Vec<CreateCommand> {
    vec![
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
                    "split",
                    "Optional delimiter to split documents on",
                ))
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
            )),
        CreateCommand::new("capacitorfile")
            .description("Manage capacitorfile recipes for this server")
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
            ),
        CreateCommand::new("train")
            .description("Train a model from a corpus or a capacitorfile recipe")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "model",
                    "Name for the new model",
                )
                .required(true),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "dataset",
                    "Name of an uploaded dataset (raw corpus) to train on",
                )
                .set_autocomplete(true),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "capacitorfile",
                    "Name of an uploaded capacitorfile recipe to train from",
                )
                .set_autocomplete(true),
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::Integer,
                "total_experts",
                "Number of experts (default 4)",
            ))
            .add_option(CreateCommandOption::new(
                CommandOptionType::Integer,
                "active_experts",
                "Active experts at query time (default 1)",
            ))
            .add_option(CreateCommandOption::new(
                CommandOptionType::Integer,
                "centroids",
                "Centroids per cluster (default 2)",
            ))
            .add_option(CreateCommandOption::new(
                CommandOptionType::Integer,
                "top_k",
                "Top-k sampling at query time (default 10)",
            ))
            .add_option(CreateCommandOption::new(
                CommandOptionType::Integer,
                "max_tokens",
                "Max tokens to generate (default 200)",
            ))
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
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::Integer,
                "seed",
                "Optional fixed random seed for reproducibility",
            )),
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
    ]
}
