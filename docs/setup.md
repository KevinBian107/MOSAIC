# Setup Environments
Always check on the [datahub.ucsd.edu/services/disk-quota-service/](https://datahub.ucsd.edu/services/disk-quota-service/) to see the quota limit we have first before installing anything. This is an instruction for creating the environment needed for this project, we will establish our environment first similar to [AutoGraph](https://github.com/BorgwardtLab/AutoGraph), then add in additional dependencies if it's needed.

> All the environment instantiation was tested on an A30 of UCSD's DSMLP system.

The following should run without an issue on DSMLP and we copied over for conviniency:

```bash
conda env create -f environment.yaml
conda activate motif-graph
```

# Setup Claude Code
We have provided the `.claude` file for helping code developement and understanding of the codebase, install claude code as the following to automatically used the codebase instructions we have provided.

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --dangerously-skip-permissions
```

To use the skills we have created, just call the keyword `investigation` and claude should be conducting reasoning and coding in the structured manner that we have defined, with logging of each step into a \scratch folder that's gitignored.