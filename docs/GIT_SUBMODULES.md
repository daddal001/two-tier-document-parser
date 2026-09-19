# Git Submodules Guide - MinerU Integration

This document explains how to use git submodules to manage the MinerU dependency in this repository.

## What Are Git Submodules?

Git submodules allow you to keep a Git repository as a subdirectory of another Git repository. This enables you to:

- ✅ **Include external code** in your repository
- ✅ **Track specific versions** of the external repository
- ✅ **Update easily** to newer versions when available
- ✅ **Maintain independence** - changes to the external repo don't automatically affect yours
- ✅ **Avoid code duplication** - reference the external repo instead of copying it

## Why Use Submodules for MinerU?

For this project, using MinerU as a submodule provides several benefits:

1. **Version Control**: Pin to a specific MinerU version that's tested and working
2. **Easy Updates**: Pull latest MinerU improvements with a simple command
3. **Clean Separation**: Your code stays separate from MinerU's code
4. **Reduced Repository Size**: GitHub stores a reference, not the full MinerU codebase
5. **Contribution Ready**: Can easily contribute improvements back to MinerU

## Initial Setup

### Step 1: Remove Current MinerU Directory (if exists)

```bash
rm -rf MinerU/
```

### Step 2: Add MinerU as a Git Submodule

```bash
git submodule add https://github.com/opendatalab/MinerU.git MinerU
```

This command:
- Clones MinerU into the `MinerU/` directory
- Creates a `.gitmodules` file tracking the submodule
- Adds a reference to the specific MinerU commit

### Step 3: Commit the Submodule

```bash
git add .gitmodules MinerU
git commit -m "Add MinerU as git submodule"
git push origin main
```

### Step 4: Install MinerU Dependencies

```bash
cd MinerU
pip install -e .[core]
mineru-models-download -s huggingface -m all
cd ..
```

## Cloning This Repository

### For New Users

**Option 1: Clone with submodules (recommended)**
```bash
git clone --recurse-submodules https://github.com/daddal001/two_tier_document_parser.git
cd two_tier_document_parser
```

**Option 2: Clone then initialize submodules**
```bash
git clone https://github.com/daddal001/two_tier_document_parser.git
cd two_tier_document_parser
git submodule init
git submodule update
```

After cloning, install MinerU:
```bash
cd MinerU
pip install -e .[core]
mineru-models-download -s huggingface -m all
cd ..
```

## Updating MinerU to Latest Version

### Update to Latest MinerU Release

```bash
# Navigate to MinerU submodule
cd MinerU

# Pull latest changes from MinerU repository
git pull origin master

# Return to project root
cd ..

# Stage the updated submodule reference
git add MinerU

# Commit the update
git commit -m "Update MinerU to latest version"

# Push to your repository
git push origin main
```

### Update to Specific MinerU Version/Tag

```bash
# Navigate to MinerU submodule
cd MinerU

# Fetch all tags
git fetch --tags

# Checkout specific version (e.g., v2.5.0)
git checkout v2.5.0

# Return to project root
cd ..

# Stage and commit
git add MinerU
git commit -m "Update MinerU to v2.5.0"
git push origin main
```

### Check Current MinerU Version

```bash
# From project root
git submodule status

# Or from within MinerU directory
cd MinerU
git log -1 --oneline
git describe --tags
```

## Common Commands Reference

### Check Submodule Status
```bash
git submodule status
```

Output example:
```
 abc1234567890def1234567890abcdef12345678 MinerU (v2.5.0)
```

### Update All Submodules to Latest
```bash
git submodule update --remote --merge
```

### Reset Submodule to Committed Version
```bash
git submodule update --init --recursive
```

### Remove Submodule (if needed)
```bash
# Remove submodule entry from .git/config
git submodule deinit -f MinerU

# Remove submodule directory from .git/modules
rm -rf .git/modules/MinerU

# Remove from working tree and .gitmodules
git rm -f MinerU
```

## GitHub Display

On GitHub, your repository will show the MinerU directory like this:

```
MinerU @ abc1234
```

Clicking on it will navigate to the specific MinerU commit referenced by your repository.

## Working with Submodules and Branches

### Creating a Branch in Your Repository

```bash
# Submodules follow along automatically
git checkout -b feature/my-feature
git push origin feature/my-feature
```

### Switching Branches

```bash
# Switch branch
git checkout main

# Update submodules to match the branch's recorded state
git submodule update --init --recursive
```

## Troubleshooting

### Issue: "MinerU directory is empty after clone"

**Solution:**
```bash
git submodule init
git submodule update
```

### Issue: "Modified content in MinerU submodule (untracked content)"

**Cause:** You have uncommitted changes inside the MinerU directory.

**Solution:**
```bash
cd MinerU
git status  # See what changed
git checkout .  # Discard changes, or
git stash  # Save changes for later
cd ..
```

### Issue: "Detached HEAD in MinerU"

**Cause:** Submodules point to specific commits, not branches (this is normal).

**Solution:** This is expected behavior. Submodules are pinned to specific commits for stability.

### Issue: "fatal: no submodule mapping found in .gitmodules"

**Cause:** .gitmodules file is missing or corrupted.

**Solution:**
```bash
# Re-add the submodule
git submodule add https://github.com/opendatalab/MinerU.git MinerU
```

### Issue: "Changes not showing up after submodule update"

**Solution:**
```bash
# Make sure you committed the submodule update
git add MinerU
git commit -m "Update MinerU submodule"
git push
```

## Best Practices

### 1. Always Commit Submodule Updates

When updating MinerU, always commit the new reference:
```bash
cd MinerU
git pull origin master
cd ..
git add MinerU
git commit -m "Update MinerU to [version/commit]"
```

### 2. Document MinerU Version in Commit Messages

Be specific about what version you're updating to:
```bash
git commit -m "Update MinerU to v2.5.1 - fixes parsing bugs"
```

### 3. Test After Updating

Always test your parser integration after updating MinerU:
```bash
cd accurate/
pytest tests/ -v
```

### 4. Pin to Stable Releases

For production, use tagged releases rather than master branch:
```bash
cd MinerU
git checkout v2.5.0  # Use stable tag
cd ..
git add MinerU
git commit -m "Pin MinerU to stable v2.5.0"
```

### 5. Include Submodule Instructions in README

(Already done - see README.md "MinerU Integration" section)

## Integration with This Project

### File Structure

```
two_tier_document_parser/
├── .gitmodules              # Submodule configuration
├── MinerU/                  # Git submodule (MinerU repository)
│   ├── .git/                # MinerU's git data
│   ├── magic_pdf/           # MinerU source code
│   └── ...
├── accurate/
│   ├── parser.py            # Uses MinerU from submodule
│   └── ...
└── ...
```

### Importing MinerU in Code

In `accurate/parser.py`:

```python
# MinerU is accessible because it's in the parent directory
import sys
sys.path.insert(0, '../MinerU')

from magic_pdf.pipe.UNIPipe import UNIPipe
# ... rest of import
```

Or install it:
```bash
cd MinerU
pip install -e .[core]
```

Then import normally:
```python
from magic_pdf.pipe.UNIPipe import UNIPipe
```

## Continuous Integration (CI)

If you use CI/CD, ensure submodules are checked out:

### GitHub Actions Example

```yaml
steps:
  - name: Checkout code with submodules
    uses: actions/checkout@v3
    with:
      submodules: recursive
```

### GitLab CI Example

```yaml
variables:
  GIT_SUBMODULE_STRATEGY: recursive
```

## Additional Resources

- [Git Submodules Official Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [MinerU Repository](https://github.com/opendatalab/MinerU)
- [MinerU Documentation](https://opendatalab.github.io/MinerU/)

## Quick Reference Card

| Task | Command |
|------|---------|
| Add submodule | `git submodule add <url> <path>` |
| Clone with submodules | `git clone --recurse-submodules <url>` |
| Initialize submodules | `git submodule init && git submodule update` |
| Update to latest | `cd MinerU && git pull origin master && cd .. && git add MinerU` |
| Check status | `git submodule status` |
| Reset submodule | `git submodule update --init --recursive` |

---

**Last Updated:** November 2025

**Related Documentation:**
- [README.md](README.md) - Project overview and quick start
- [Contributing](../CONTRIBUTING.md) - Development and contribution guidance
- [Changelog](../CHANGELOG.md) - Release history
