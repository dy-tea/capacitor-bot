/// Format a byte count as a human-readable string.
pub(crate) fn format_size(size: f64) -> String {
    if size > 1024.0 * 1024.0 * 1024.0 {
        format!("{:.2} GB", size / 1024.0 / 1024.0 / 1024.0)
    } else if size > 1024.0 * 1024.0 {
        format!("{:.2} MB", size / 1024.0 / 1024.0)
    } else if size > 1024.0 {
        format!("{:.2} KB", size / 1024.0)
    } else {
        format!("{} B", size)
    }
}

/// Truncate `text` to at most `max` unicode scalar values, appending a marker
/// when truncation occurs.
pub(crate) fn clip(text: &str, max: usize) -> String {
    let mut s: String = text.chars().take(max).collect();

    if text.len() > s.len() {
        s.push_str("\n... (truncated)");
    }

    s
}
