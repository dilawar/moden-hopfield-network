//! test meena cli.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

const CLI_NAME: &str = "modern-hopfield-network";

#[test]
fn cli_help() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin(CLI_NAME)?;
    cmd.arg("help");
    cmd.assert().success();
    Ok(())
}

#[test]
fn cli_xor() -> Result<(), Box<dyn std::error::Error>> {
    const DAM_NAME: &str = "xor";
    let mut cmd = Command::cargo_bin(CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "clear"]);
    cmd.assert().success();

    let mut cmd = Command::cargo_bin(CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "0,0", "0", "2"]);
    cmd.assert().success();

    cmd = Command::cargo_bin(CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "0,1", "1", "2"]);
    cmd.assert().success();

    cmd = Command::cargo_bin(CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "1,0", "1", "2"]);
    cmd.assert().success();

    cmd = Command::cargo_bin(CLI_NAME)?;
    cmd.args(["--name", DAM_NAME, "train", "1,1", "0", "2"]);
    cmd.assert().success();

    for (pat, res) in [("0,0", "0"), ("0,1", "1"), ("1,0", "1"), ("1,1", "0")] {
        cmd = Command::cargo_bin(CLI_NAME)?;
        cmd.args(["--name", DAM_NAME, "classify", pat]);
        cmd.assert().success().stdout(predicate::str::contains(res));
    }

    Ok(())
}
