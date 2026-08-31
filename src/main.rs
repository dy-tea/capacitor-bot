use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serenity::all::*;

use crate::bot::{Bot, Data};
use crate::jobs::Trainer;
use crate::query::QueryEngine;
use crate::store::Store;

mod bot;
mod jobs;
mod query;
mod recipe_edit;
mod store;
mod util;

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> anyhow::Result<T> {
    Ok(match std::env::var(name) {
        Ok(raw) => raw
            .parse::<T>()
            .map_err(|_| anyhow::anyhow!("invalid value for {name}: {raw}"))?,
        Err(_) => default,
    })
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

    let data = Arc::new(Data::new(
        store,
        Trainer::spawn(train_workers),
        QueryEngine::new(query_capacity, cache_size),
    ));

    let mut client = Client::builder(
        &token,
        GatewayIntents::non_privileged() | GatewayIntents::MESSAGE_CONTENT,
    )
    .event_handler(Bot {
        data: Arc::clone(&data),
    })
    .await
    .map_err(anyhow::Error::from)?;

    client.start().await?;

    Ok(())
}
