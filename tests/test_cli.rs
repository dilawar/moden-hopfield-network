//! test meena cli.

use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs

const MEENA_CLI_NAME: &str = "libmeena";

#[test]
fn cli_help() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
    cmd.arg("help");
    cmd.assert().success();
    Ok(())
}

#[test]
fn cli_xor() -> Result<(), Box<dyn std::error::Error>> {
    const DAM_NAME: &str = "xor";
    let mut cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "clear"]);
    cmd.assert().success();

    let mut cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "0,0", "0", "2"]);
    cmd.assert().success();

    cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "0,1", "1", "2"]);
    cmd.assert().success();

    cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "1,0", "1", "2"]);
    cmd.assert().success();

    cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "1,1", "0", "2"]);
    cmd.assert().success();

    for (pat, res) in [("0,0", "0"), ("0,1", "1"), ("1,0", "1"), ("1,1", "0")] {
        cmd = Command::cargo_bin(MEENA_CLI_NAME)?;
        cmd.args(["--name", DAM_NAME, "classify", pat]);
        cmd.assert().success().stdout(predicate::str::contains(res));
    }

    Ok(())
}
