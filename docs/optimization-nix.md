# Optimization Strategies for Fast Nix Flakes

## Building Fast Flakes

### 1. **Leverage Binary Caches**
- Use Cachix or other binary cache services
- Configure `substituters` (remote binary caches that provide pre-built packages) in your flake to avoid rebuilding packages
- Consider setting up your own binary cache for CI/CD

### 2. **Minimize Dependencies**
- Only include necessary inputs in your flake
- Avoid deep dependency chains
- Use `follows` to deduplicate transitive dependencies
    - `follows` allows you to override a flake input's dependency to use the same version as another input
    - Prevents multiple versions of the same package in your dependency tree
    - Example syntax in `flake.nix`:
        ```nix
        inputs = {
            nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
            flake-utils.url = "github:numtide/flake-utils";
            # Make flake-utils use the same nixpkgs
            flake-utils.inputs.nixpkgs.follows = "nixpkgs";
        };
        ```
    - Reduces closure size and eliminates duplicate builds
    - Particularly useful when multiple inputs depend on `nixpkgs`

### 3. **Use Shallow Git Clones**
- For flake inputs, use `shallow = true` where possible
- Reduces download time and disk usage
- Add to your `flake.nix` inputs like this:
    ```nix
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        some-input = {
            url = "github:owner/repo";
            flake = false;  # Optional: if input isn't a flake
            shallow = true;  # Enable shallow clone
        };
    };
    ```
- Can also specify depth: `shallow = true` defaults to depth 1
- Works with any Git-based input (GitHub, GitLab, self-hosted, etc.)


**Advantages:**
- Faster initial clone times (only fetches recent history)
- Lower disk space usage
- Reduced network bandwidth consumption
- Ideal for CI/CD environments where full history isn't needed

**Disadvantages:**
- May cause issues with some Git operations (e.g., `git blame` with deep history)
- Can break if the repository is rebased or force-pushed
- Not suitable when you need access to full commit history
- Some flake operations might require fetching additional history later
- May not work well with all flake inputs (some packages reference old commits)

### 4. **Enable Parallel Builds**
- Set `max-jobs` and `cores` in `nix.conf` (typically located at `/etc/nix/nix.conf` or `~/.config/nix/nix.conf`)
- Add these lines to your `nix.conf`:
    ```conf
    max-jobs = auto
    cores = 0
    ```
- `max-jobs = auto` uses the number of CPU cores available
- `cores = 0` means each build can use all available cores
- After modifying `nix.conf`, restart the Nix daemon: `sudo systemctl restart nix-daemon`

### 5. **Lock Your Inputs**
- Commit `flake.lock` to version control
- Prevents unnecessary re-evaluation and ensures reproducibility
- Update inputs intentionally with `nix flake update`

### 6. **Optimize Evaluation**
- Use `builtins.fetchTree` for large repositories
- Minimize IFD (Import From Derivation)
- Cache evaluation results when possible

### 7. **Profile and Measure**
- Use `nix build --profile` to identify bottlenecks
- Monitor build times with `nix build --print-build-logs`
- Use `nix why-depends` to understand dependency chains

### 8. **Content-Addressed Derivations**
- Enable `ca-derivations` experimental feature for better caching
- Reduces rebuilds when inputs change but outputs remain the same