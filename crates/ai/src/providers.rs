//! Providers we know by name: where they answer, which environment variable holds the key, and the
//! model a new backend starts on. One table, so detection, key lookup and the UI agree.

use crate::BackendKind;

/// One hosted provider, as a row of data rather than a match arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provider {
    /// The name a backend is added under.
    pub name: &'static str,
    pub kind: BackendKind,
    /// The OpenAI-compatible base URL, or empty for a native API.
    pub base_url: &'static str,
    /// The environment variable that conventionally holds its key.
    pub key_env: &'static str,
    /// A second name some tools use for the same key.
    pub alt_key_env: Option<&'static str>,
    /// The model a new backend starts on, before the provider's own list is fetched.
    pub default_model: &'static str,
    /// Its id in the shared models.dev catalogue, where it differs from ours.
    pub catalog_id: &'static str,
    /// A second host the same service documents, when it has one.
    pub alt_base_url: Option<&'static str>,
}

/// The priced APIs we can talk to over an OpenAI-compatible endpoint. Pay-as-you-go services are
/// offered only when their key is already set, like OpenAI and Anthropic, so nobody is advertised
/// a service they cannot use.
pub const PROVIDERS: &[Provider] = &[
    Provider {
        name: "Moonshot",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.moonshot.ai/v1",
        key_env: "MOONSHOT_API_KEY",
        alt_key_env: None,
        default_model: "kimi-k3.1-pro",
        catalog_id: "moonshotai",
        alt_base_url: None,
    },
    Provider {
        name: "DeepSeek",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.deepseek.com/v1",
        key_env: "DEEPSEEK_API_KEY",
        alt_key_env: None,
        default_model: "deepseek-flash",
        catalog_id: "deepseek",
        alt_base_url: None,
    },
    Provider {
        name: "xAI (Grok)",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.x.ai/v1",
        key_env: "XAI_API_KEY",
        alt_key_env: None,
        default_model: "grok-4.7",
        catalog_id: "xai",
        alt_base_url: None,
    },
    Provider {
        name: "Groq",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.groq.com/openai/v1",
        key_env: "GROQ_API_KEY",
        alt_key_env: None,
        default_model: "openai/gpt-oss-120b",
        catalog_id: "groq",
        alt_base_url: None,
    },
    Provider {
        name: "Mistral",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.mistral.ai/v1",
        key_env: "MISTRAL_API_KEY",
        alt_key_env: None,
        default_model: "mistral-large-4",
        catalog_id: "mistral",
        alt_base_url: None,
    },
    Provider {
        name: "Together",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.together.xyz/v1",
        key_env: "TOGETHER_API_KEY",
        alt_key_env: None,
        default_model: "deepseek-ai/DeepSeek-V4.1-Flash",
        catalog_id: "togetherai",
        alt_base_url: Some("https://api.together.ai/v1"),
    },
    Provider {
        name: "Fireworks",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.fireworks.ai/inference/v1",
        key_env: "FIREWORKS_API_KEY",
        alt_key_env: None,
        default_model: "accounts/fireworks/models/gpt-oss-120b",
        catalog_id: "fireworks-ai",
        alt_base_url: None,
    },
    Provider {
        name: "DeepInfra",
        kind: BackendKind::OpenaiCompatible,
        base_url: "https://api.deepinfra.com/v1/openai",
        key_env: "DEEPINFRA_API_KEY",
        alt_key_env: None,
        default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        catalog_id: "deepinfra",
        alt_base_url: None,
    },
];

impl Provider {
    /// The key names this provider accepts, in order.
    pub fn key_names(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.key_env).chain(self.alt_key_env)
    }
}

/// Split a base URL into scheme, host and path, lowercasing the host.
fn split(base: &str) -> Option<(&str, String, &str)> {
    let (scheme, rest) = base.trim().split_once("://")?;
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    Some((scheme, host.to_ascii_lowercase(), path))
}

/// The provider whose base URL this is, if any.
///
/// HTTPS only: a key must never be handed to a plain-HTTP base URL that a settings file could be
/// edited into. A pasted base without the trailing `/v1` still finds its row.
pub fn of_base(base: &str) -> Option<&'static Provider> {
    let (scheme, host, path) = split(base.trim().trim_end_matches('/'))?;
    if !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    let path = path.trim_end_matches('/');
    PROVIDERS.iter().find(|p| {
        [Some(p.base_url), p.alt_base_url].into_iter().flatten().any(|url| {
            let Some((_, name, theirs)) = split(url) else { return false };
            let theirs = theirs.trim_matches('/');
            name == host && (path == theirs || path.is_empty() || format!("{theirs}/v1") == path || format!("{path}/v1") == theirs)
        })
    })
}

/// The key this provider holds in the environment, when one is set. Blank counts as unset.
pub fn env_key(base: &str) -> Option<String> {
    env_key_with(base, |name| std::env::var(name).ok())
}

/// The same, with the environment passed in, so tests need no process state.
pub fn env_key_with(base: &str, get: impl Fn(&str) -> Option<String>) -> Option<String> {
    let provider = of_base(base)?;
    provider.key_names().filter_map(get).find(|k| !k.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row has what a backend needs, and the names are unique.
    #[test]
    fn every_provider_is_complete_and_named_once() {
        for p in PROVIDERS {
            assert!(!p.name.trim().is_empty());
            assert!(!p.key_env.trim().is_empty(), "{} has no key variable", p.name);
            assert!(!p.default_model.trim().is_empty(), "{} has no default model", p.name);
            assert!(!p.catalog_id.trim().is_empty(), "{} has no catalogue id", p.name);
            if p.kind == BackendKind::OpenaiCompatible {
                assert!(p.base_url.starts_with("https://"), "{} is not https", p.name);
            }
        }
        let mut names: Vec<&str> = PROVIDERS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PROVIDERS.len(), "two providers share a name");
    }

    /// Two rows must never share a key or a base: one key would be sent to two services.
    #[test]
    fn keys_and_bases_are_unique() {
        let mut keys: Vec<&str> = PROVIDERS.iter().map(|p| p.key_env).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), PROVIDERS.len(), "two providers share a key variable");

        let mut bases: Vec<&str> = PROVIDERS.iter().map(|p| p.base_url).collect();
        bases.sort_unstable();
        bases.dedup();
        assert_eq!(bases.len(), PROVIDERS.len(), "two providers share a base URL");

        let mut catalog: Vec<&str> = PROVIDERS.iter().map(|p| p.catalog_id).collect();
        catalog.sort_unstable();
        catalog.dedup();
        assert_eq!(catalog.len(), PROVIDERS.len(), "two providers share a catalogue id");
    }

    /// Together documents two hosts; both find its row, so its key is not lost on either.
    #[test]
    fn a_documented_second_host_finds_its_provider() {
        let by_xyz = of_base("https://api.together.xyz/v1").map(|p| p.name);
        let by_ai = of_base("https://api.together.ai/v1").map(|p| p.name);
        assert_eq!(by_xyz, Some("Together"));
        assert_eq!(by_ai, Some("Together"));
    }

    /// A plain-HTTP base never receives a key, and a base missing `/v1` still finds its row.
    #[test]
    fn only_https_bases_are_recognised() {
        assert!(of_base("http://api.deepseek.com/v1").is_none());
        assert_eq!(of_base("https://api.deepseek.com"), Some(&PROVIDERS[1]));
        assert_eq!(of_base("HTTPS://API.DEEPSEEK.COM/v1"), Some(&PROVIDERS[1]));
    }

    /// A blank variable counts as unset, so the next name is tried.
    #[test]
    fn a_blank_key_falls_through_to_the_next_name() {
        let base = "https://api.deepseek.com/v1";
        let empty = env_key_with(base, |_| Some("  ".into()));
        assert_eq!(empty, None);
        let found = env_key_with(base, |name| (name == "DEEPSEEK_API_KEY").then(|| "k".to_string()));
        assert_eq!(found.as_deref(), Some("k"));
    }

    /// A base URL finds its row, with or without a trailing slash; someone else's doesn't.
    #[test]
    fn a_base_url_finds_its_provider() {
        assert_eq!(of_base("https://api.deepseek.com/v1"), Some(&PROVIDERS[1]));
        assert_eq!(of_base("https://api.deepseek.com/v1/"), Some(&PROVIDERS[1]));
        assert_eq!(of_base("http://127.0.0.1:11434/v1"), None);
        assert_eq!(of_base("https://api.example.test/v1"), None);
    }
}
