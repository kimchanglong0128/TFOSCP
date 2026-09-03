cd /workspace && source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
# restore after pod restart (container disk is wiped on stop)
[ -e ~/.claude ]      || ln -s /workspace/.claude ~/.claude
[ -e ~/.local ]       || ln -s /workspace/.claude_local ~/.local
[ -e ~/.claude.json ] || ln -s /workspace/.claude_json ~/.claude.json
[ -e ~/.config ]      || ln -s /workspace/.config ~/.config
export PATH=$HOME/.local/bin:/workspace/bin:$PATH

# gh (container disk is wiped on stop; reinstall if missing)
command -v gh >/dev/null || {
  mkdir -p -m 755 /etc/apt/keyrings
  wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg > /etc/apt/keyrings/githubcli-archive-keyring.gpg
  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
  apt update -qq && apt install gh -y -qq
}

# github (credentials live in ~/.config/gh on the volume, not here)
git config --global user.name "kimchanglong0128"
git config --global user.email "kimchanglong0128@gmail.com"
gh auth setup-git 2>/dev/null
