# GitHub Setup Guide for GlassVAE

## Step 1: Configure Git (if not already done)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 2: Choose Authentication Method

### Option A: SSH (Recommended - More Secure)

1. **Check if you have SSH keys:**
   ```bash
   ls -la ~/.ssh/id_*.pub
   ```

2. **If no SSH key exists, generate one:**
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # Press Enter to accept default location
   # Optionally set a passphrase for extra security
   ```

3. **Copy your public key:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   # Copy the entire output
   ```

4. **Add SSH key to GitHub:**
   - Go to GitHub.com → Settings → SSH and GPG keys
   - Click "New SSH key"
   - Paste your public key
   - Give it a title and save

5. **Test SSH connection:**
   ```bash
   ssh -T git@github.com
   # Should see: "Hi username! You've successfully authenticated..."
   ```

### Option B: HTTPS (Easier, but requires token)

1. **Use Personal Access Token:**
   - Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` scope
   - Copy the token (you'll need it when pushing)

## Step 3: Initialize Git Repository

```bash
cd /home/qchen363/GlassVAE
git init
git add .
git commit -m "Initial commit: GlassVAE project structure"
```

## Step 4: Create GitHub Repository

1. Go to GitHub.com
2. Click "+" → "New repository"
3. Name it: `GlassVAE` (or your preferred name)
4. **Don't** initialize with README, .gitignore, or license (we already have files)
5. Click "Create repository"

## Step 5: Connect Local Repository to GitHub

### If using SSH:
```bash
git remote add origin git@github.com:YOUR_USERNAME/GlassVAE.git
git branch -M main
git push -u origin main
```

### If using HTTPS:
```bash
git remote add origin https://github.com/YOUR_USERNAME/GlassVAE.git
git branch -M main
git push -u origin main
# When prompted, use your GitHub username and Personal Access Token as password
```

## Step 6: Verify Connection

```bash
git remote -v
# Should show your GitHub repository URL
```

## Common Commands

```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull from GitHub
git pull

# View remote
git remote -v
```

## Troubleshooting

### If SSH connection fails:
- Make sure your SSH key is added to GitHub
- Test with: `ssh -T git@github.com`
- Check SSH agent: `eval "$(ssh-agent -s)"` then `ssh-add ~/.ssh/id_ed25519`

### If HTTPS push fails:
- Make sure you're using Personal Access Token, not password
- Token needs `repo` scope

### If you need to change remote URL:
```bash
git remote set-url origin NEW_URL
```

