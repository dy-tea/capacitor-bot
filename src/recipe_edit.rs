//! Live recipe editing.
//!
//! `/recipe create` and `/recipe edit` both prompt the user to reply to a bot
//! message with recipe text. The reply's text is saved to disk, and an
//! in-memory session keyed on that message's id is registered so that later
//! edits to the reply keep the on-disk recipe file in sync (see
//! [`sync_edited_message`], called from the `message_update` event handler).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serenity::all::*;

use crate::util::{clip, format_size};

/// How long a `/recipe create` or `/recipe edit` command waits for the user to
/// reply with the recipe content before giving up.
pub(crate) const REPLY_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// A recipe file being kept in sync with a replied-to Discord message.
///
/// When a user replies to a recipe prompt, that reply's text is saved to
/// `path`. While the session exists, subsequent edits to the reply rewrite the
/// file, and `confirm` (the bot's acknowledgement message) is refreshed to
/// reflect the new size.
#[derive(Clone)]
pub(crate) struct RecipeSession {
    pub path: PathBuf,
    pub name: String,
    pub confirm: Option<(ChannelId, MessageId)>,
}

/// Maps the id of a recipe reply to the file it is synced to.
pub(crate) type RecipeSessions = Arc<Mutex<HashMap<MessageId, RecipeSession>>>;

/// Create a fresh session registry for a bot instance.
pub(crate) fn new_sessions() -> RecipeSessions {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Posts `prompt` as the interaction's response and waits for a reply to it
/// from the invoking user. Returns the reply message, or `None` on timeout.
pub(crate) async fn await_reply(
    ctx: &Context,
    cmd: &CommandInteraction,
    prompt: String,
) -> anyhow::Result<Option<Message>> {
    cmd.create_response(
        ctx,
        CreateInteractionResponse::Message(
            CreateInteractionResponseMessage::new().content(clip(&prompt, 2000)),
        ),
    )
    .await?;

    let prompt_msg = cmd.get_response(ctx).await?;
    let prompt_id = prompt_msg.id;

    let collector = MessageCollector::new(ctx)
        .timeout(REPLY_TIMEOUT)
        .channel_id(prompt_msg.channel_id)
        .author_id(cmd.user.id)
        .filter(move |m: &Message| {
            m.message_reference
                .as_ref()
                .and_then(|r| r.message_id)
                .is_some_and(|id| id == prompt_id)
        });

    Ok(collector.next().await)
}

/// Posts a confirmation for a recipe saved from `reply`, then registers a
/// live-edit session keyed on `reply.id` so `message_update` can keep `path`
/// in sync as the user edits their reply.
pub(crate) async fn register_session(
    ctx: &Context,
    cmd: &CommandInteraction,
    sessions: &RecipeSessions,
    reply: &Message,
    path: PathBuf,
    content: &str,
) -> anyhow::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("capacitorfile")
        .to_string();

    let size = format_size(content.len() as f64);

    let confirm = cmd
        .channel_id
        .send_message(
            ctx,
            CreateMessage::new().content(format!(
                "Saved recipe `{file_name}` ({size}). Editing your reply updates the file."
            )),
        )
        .await?;

    let mut sessions = sessions.lock().unwrap();
    sessions.insert(
        reply.id,
        RecipeSession {
            path,
            name: file_name,
            confirm: Some((confirm.channel_id, confirm.id)),
        },
    );

    Ok(())
}

/// Handle a `message_update` event for the live recipe flow: if the edited
/// message is a tracked recipe reply, rewrite the recipe file with the new text
/// and refresh the bot's confirmation message.
pub(crate) async fn sync_edited_message(
    ctx: &Context,
    sessions: &RecipeSessions,
    event: &MessageUpdateEvent,
) {
    let Some(content) = &event.content else {
        return;
    };

    let session = {
        let sessions = sessions.lock().unwrap();
        sessions.get(&event.id).cloned()
    };

    let Some(session) = session else {
        return;
    };

    let content = strip_fence(&content);

    if let Err(err) = std::fs::write(&session.path, content.as_bytes()) {
        eprintln!(
            "failed to sync edited recipe into {path:?}: {err}",
            path = session.path
        );
        return;
    }

    let size = format_size(content.len() as f64);
    let confirm_text = format!(
        "Saved recipe `{name}` ({size}). Editing your reply updates the file.",
        name = session.name,
    );

    let Some((channel_id, msg_id)) = session.confirm else {
        return;
    };

    if let Err(err) = channel_id
        .edit_message(
            ctx,
            msg_id,
            EditMessage::new().content(clip(&confirm_text, 2000)),
        )
        .await
    {
        eprintln!("failed to refresh recipe confirm message: {err}");
    }
}

/// Strip a single layer of surrounding code fences (```` ``` ````) from recipe
/// text that was pasted into a reply, so the saved file contains the recipe
/// text itself rather than the markdown wrapper.
pub(crate) fn strip_fence(content: &str) -> String {
    // Drop trailing blank/whitespace lines so a stray newline after the closing
    // fence doesn't break detection.
    let trimmed = content.trim_end();
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.len() >= 2
        && lines[0].trim_start().starts_with("```")
        && lines[lines.len() - 1].trim_start().starts_with("```")
    {
        // Drop the opening/closing fence lines (the opening one may carry a
        // language tag after the backticks, which is discarded here).
        lines[1..lines.len() - 1].join("\n")
    } else {
        content.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::strip_fence;

    #[test]
    fn leaves_unfenced_text_untouched() {
        let input = "name: foo\ndatasets:\n  - data.txt".to_string();
        assert_eq!(strip_fence(&input), input);
    }

    #[test]
    fn strips_a_plain_code_fence() {
        let input = "```\nname: foo\nkey: value\n```";
        assert_eq!(strip_fence(input), "name: foo\nkey: value");
    }

    #[test]
    fn strips_a_fence_with_language_tag() {
        let input = "```capacitor\nname: foo\nkey: value\n```";
        assert_eq!(strip_fence(input), "name: foo\nkey: value");
    }

    #[test]
    fn strips_a_fence_with_trailing_spaces_and_newlines() {
        let input = "```\nname: foo\nkey: value\n```\n\n";
        assert_eq!(strip_fence(input), "name: foo\nkey: value");
    }

    #[test]
    fn does_not_touch_inline_fences() {
        let input = "some ``` inline ``` text";
        assert_eq!(strip_fence(input), input);
    }

    #[test]
    fn leaves_single_line_untouched() {
        // A single line starting with backticks but with no closing line is not
        // treated as a fence pair.
        let input = "```name: foo";
        assert_eq!(strip_fence(input), input);
    }
}
