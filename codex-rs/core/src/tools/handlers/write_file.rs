//! Handler for the synthetic `write_file` tool exposed to local models.
//!
//! Writes (creates or overwrites) a file directly. The point of having a real
//! handler — rather than translating `write_file` to a `shell` `printf` — is
//! that the model then sees its *own* `write_file` call in the transcript. When
//! it was translated to shell, the model looked back at a `printf '%s' '<single-
//! quote-escaped content>'` command, decided "the shell escaping mangled my
//! file," and looped trying every other write method. The file was always fine;
//! the model just couldn't tell. A dedicated tool keeps its mental model intact.

use std::path::Path;

use codex_utils_absolute_path::AbsolutePathBuf;
use serde::Deserialize;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub struct WriteFileHandler;

#[derive(Deserialize)]
struct WriteFileArgs {
    #[serde(alias = "file", alias = "file_path", alias = "filename")]
    path: String,
    #[serde(default, alias = "contents", alias = "text", alias = "body")]
    content: String,
}

impl ToolHandler for WriteFileHandler {
    type Output = FunctionToolOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<Self::Output, FunctionCallError> {
        let ToolInvocation { turn, payload, .. } = invocation;
        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "write_file handler received unsupported payload".to_string(),
                ));
            }
        };
        let args: WriteFileArgs = parse_arguments(&arguments)?;
        let path = args.path.trim();
        if path.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "write_file: path must not be empty".to_string(),
            ));
        }
        let abs = AbsolutePathBuf::resolve_path_against_base(Path::new(path), &turn.cwd);
        // Same sandbox gate apply_patch uses: refuse writes the policy disallows.
        if !turn
            .file_system_sandbox_policy
            .can_write_path_with_cwd(abs.as_path(), turn.cwd.as_path())
        {
            return Err(FunctionCallError::RespondToModel(format!(
                "write_file: '{path}' is outside the writable sandbox — cannot write there."
            )));
        }
        if let Some(parent) = abs.as_path().parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return Err(FunctionCallError::RespondToModel(format!(
                "write_file: could not create parent directory: {e}"
            )));
        }
        let bytes = args.content.as_bytes();
        if let Err(e) = tokio::fs::write(abs.as_path(), bytes).await {
            return Err(FunctionCallError::RespondToModel(format!(
                "write_file: failed to write {path}: {e}"
            )));
        }
        Ok(FunctionToolOutput::from_text(
            format!("Wrote {} bytes to {path}", bytes.len()),
            Some(true),
        ))
    }
}
