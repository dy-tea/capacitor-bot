use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Registry metadata for a single trained model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub name: String,
    /// Absolute path to the compiled model binary.
    pub path: PathBuf,
    /// Absolute paths to the dataset files the model was trained on.
    pub datasets: Vec<PathBuf>,
    pub owner: u64,
    pub created_at: u64,
    /// The recipe text used to build the model (for reproducibility).
    pub recipe: String,
}

/// Per-namespace (guild id, or user id fallback) model list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    pub namespaces: HashMap<u64, Vec<ModelMeta>>,
    /// Namespace -> name of the most recently queried model.
    #[serde(default)]
    pub last_used: HashMap<u64, String>,
}

/// On-disk persistence for uploaded datasets, compiled models and the registry.
pub struct Store {
    root: PathBuf,
    registry: Registry,
}

impl Store {
    /// Open (or create) the store rooted at `root`, loading the registry if present.
    pub fn new(root: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&root)?;

        let registry_path = root.join("registry.json");

        let registry = if registry_path.exists() {
            let raw = std::fs::read_to_string(&registry_path)?;
            serde_json::from_str(&raw)?
        } else {
            Registry::default()
        };

        Ok(Self { root, registry })
    }

    fn namespace_dir(&self, namespace: u64) -> PathBuf {
        self.root.join(namespace.to_string())
    }

    pub fn datasets_dir(&self, namespace: u64) -> PathBuf {
        self.namespace_dir(namespace).join("datasets")
    }

    pub fn models_dir(&self, namespace: u64) -> PathBuf {
        self.namespace_dir(namespace).join("models")
    }

    // -- Datasets ---------------------------------------------------------

    /// Persist an uploaded attachment as a dataset and return its absolute path.
    pub fn save_dataset(
        &self,
        namespace: u64,
        filename: &str,
        bytes: &[u8],
    ) -> anyhow::Result<PathBuf> {
        let dir = self.datasets_dir(namespace);
        std::fs::create_dir_all(&dir)?;

        let safe = sanitize_filename(filename);
        let path = unique_path(dir.join(&safe));

        std::fs::write(&path, bytes)?;

        Ok(path)
    }

    /// Load a dataset path from the given name (exact, sanitized). Returns `None`
    /// if no matching dataset exists.
    pub fn find_dataset(&self, namespace: u64, name: &str) -> Option<PathBuf> {
        let desired = sanitize_filename(name);
        let dir = self.datasets_dir(namespace);

        let entries = std::fs::read_dir(&dir).ok()?;

        for entry in entries.flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };

            if file_name == desired
                || file_name.strip_suffix(".dataset").map(str::to_owned) == Some(desired.clone())
            {
                return Some(path);
            }
        }

        None
    }

    /// List the dataset files uploaded for a namespace.
    pub fn list_datasets(&self, namespace: u64) -> Vec<PathBuf> {
        let dir = self.datasets_dir(namespace);

        std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .collect()
    }

    /// Record the name of the most recently queried model for a namespace.
    pub fn set_last_used(&mut self, namespace: u64, name: &str) -> anyhow::Result<()> {
        self.registry.last_used.insert(namespace, name.to_string());
        self.persist()
    }

    /// Name of the most recently queried model for a namespace, if any.
    pub fn last_used(&self, namespace: u64) -> Option<String> {
        self.registry.last_used.get(&namespace).cloned()
    }

    // -- Registry ---------------------------------------------------------

    fn persist(&self) -> anyhow::Result<()> {
        let registry_path = self.root.join("registry.json");
        let raw = serde_json::to_string_pretty(&self.registry)?;

        std::fs::write(registry_path, raw)?;

        Ok(())
    }

    fn models_of(&mut self, namespace: u64) -> &mut Vec<ModelMeta> {
        self.registry.namespaces.entry(namespace).or_default()
    }

    /// Register a freshly built model and persist the registry.
    pub fn register_model(&mut self, namespace: u64, meta: ModelMeta) -> anyhow::Result<()> {
        let models = self.models_of(namespace);

        if let Some(existing) = models.iter_mut().find(|m| m.name == meta.name) {
            existing.datasets = meta.datasets;
            existing.owner = meta.owner;
            existing.created_at = meta.created_at;
            existing.recipe = meta.recipe;
            existing.path = meta.path;
        } else {
            models.push(meta);
        }

        self.persist()
    }

    /// List models for a namespace.
    pub fn list(&self, namespace: u64) -> Vec<ModelMeta> {
        self.registry
            .namespaces
            .get(&namespace)
            .cloned()
            .unwrap_or_default()
    }

    /// Look up a model by name for a namespace.
    pub fn get(&self, namespace: u64, name: &str) -> Option<ModelMeta> {
        self.registry
            .namespaces
            .get(&namespace)?
            .iter()
            .find(|m| m.name == name)
            .cloned()
    }

    /// Remove a model (registry entry + on-disk binary) for a namespace.
    /// Returns `true` if an entry was removed.
    pub fn delete(&mut self, namespace: u64, name: &str) -> anyhow::Result<bool> {
        let Some(models) = self.registry.namespaces.get_mut(&namespace) else {
            return Ok(false);
        };

        let Some(index) = models.iter().position(|m| m.name == name) else {
            return Ok(false);
        };

        let meta = models.remove(index);

        if meta.path.exists() {
            std::fs::remove_file(&meta.path)?;
        }

        self.persist()?;

        Ok(true)
    }
}

/// Derive a filesystem-safe name from a raw filename.
fn sanitize_filename(name: &str) -> String {
    let name = name.trim();

    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_ascii_punctuation() {
                c
            } else {
                '_'
            }
        })
        .collect();

    let cleaned = cleaned.trim_matches(['.', '_']);

    if cleaned.is_empty() {
        String::from("dataset")
    } else {
        cleaned.to_string()
    }
}

/// Append a numeric suffix until the path is free.
fn unique_path(path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string();
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut i = 1;

    loop {
        let candidate = parent.join(format!("{file_name}.{i}"));
        if !candidate.exists() {
            return candidate;
        }
        i += 1;
    }
}
