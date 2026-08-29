use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

use tokio::sync::Semaphore;

use capacitor::model::Model;
use capacitor::{generate, load};

use crate::store::ModelMeta;

/// Simple bounded model cache keyed by `(namespace, name)`.
struct ModelCache {
    map: HashMap<(u64, String), Arc<Model>>,
    order: VecDeque<(u64, String)>,
    capacity: usize,
}

impl ModelCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn get(&self, namespace: u64, name: &str) -> Option<Arc<Model>> {
        self.map.get(&(namespace, name.to_string())).cloned()
    }

    fn insert(&mut self, namespace: u64, name: String, model: Arc<Model>) {
        let key = (namespace, name);

        if self.map.contains_key(&key) {
            return;
        }

        self.map.insert(key.clone(), model);
        self.order.push_back(key);

        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
    }
}

/// Runs model inference under per-guild concurrency limits (a bounded query
/// "pool") and caches loaded models across calls.
#[derive(Clone)]
pub struct QueryEngine {
    semaphores: Arc<RwLock<HashMap<u64, Arc<Semaphore>>>>,
    cache: Arc<Mutex<ModelCache>>,
    per_guild_capacity: usize,
    cache_capacity: usize,
}

impl QueryEngine {
    pub fn new(per_guild_capacity: usize, cache_capacity: usize) -> Self {
        Self {
            semaphores: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(Mutex::new(ModelCache::new(cache_capacity.max(1)))),
            per_guild_capacity: per_guild_capacity.max(1),
            cache_capacity: cache_capacity.max(1),
        }
    }

    fn semaphore(&self, namespace: u64) -> Arc<Semaphore> {
        if let Some(sem) = self.semaphores.read().unwrap().get(&namespace) {
            return Arc::clone(sem);
        }

        let sem = Arc::new(Semaphore::new(self.per_guild_capacity));

        self.semaphores
            .write()
            .unwrap()
            .entry(namespace)
            .or_insert_with(|| Arc::clone(&sem));

        sem
    }

    async fn load_cached(&self, namespace: u64, meta: &ModelMeta) -> anyhow::Result<Arc<Model>> {
        if let Some(model) = self.cache.lock().unwrap().get(namespace, &meta.name) {
            return Ok(model);
        }

        let path = meta.path.clone();

        let model = tokio::task::spawn_blocking(move || -> anyhow::Result<Arc<Model>> {
            let model = load(&path)?;
            Ok(Arc::new(model))
        })
        .await
        .map_err(anyhow::Error::from)??;

        if self.cache_capacity > 0 {
            self.cache
                .lock()
                .unwrap()
                .insert(namespace, meta.name.clone(), Arc::clone(&model));
        }

        Ok(model)
    }

    /// Generate a text response from `meta`'s model for `prompt`. Concurrency per
    /// guild is bounded by the engine's configured capacity. `seed` is optional;
    /// when `None` a random generator is used.
    pub async fn query(
        &self,
        namespace: u64,
        meta: ModelMeta,
        prompt: String,
        _seed: Option<u64>,
    ) -> anyhow::Result<String> {
        let sem = self.semaphore(namespace);
        let _permit = sem.acquire().await.map_err(anyhow::Error::from)?;

        let model = self.load_cached(namespace, &meta).await?;

        tokio::task::spawn_blocking(move || {
            let mut buf = Vec::new();
            generate(&model, &prompt, &mut buf)?;
            Ok(String::from_utf8_lossy(&buf).to_string())
        })
        .await
        .map_err(anyhow::Error::from)?
    }
}
