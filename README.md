# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to get started with classical medical image analysis.

During the MIALab you will work on the task of brain tissue segmentation from magnetic resonance (MR) images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification
- Post-processing
- Evaluation

After you complete the exercises, dive into the 
    
    pipeline.py 

script to learn how all of these steps work together. 

During the laboratory you will get to know the entire pipeline and investigate one of these pipeline elements in-depth.
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer or researcher in the medical image analysis domain.

Enjoy!

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.

# 102475-HS2024-0: Medical Image Analysis Lab

This is the repository for our MIA Lab pipeline-project submission.

### Group Members
José Miguel PINTO INÁCIO | jose.inacio@students.unibe.ch

Marco Daniel PORTMANN | marco.portmann@students.unibe.ch

## Useful links

-   [MIA24 Slack](mialab2024.slack.com)

## Git Basics

### Cloning the Repository
To start working with this repository, clone it to your local machine using HTTPS, which requires a personal access token (classic).

1. **Generate a Personal Access Token**: Go to your GitHub account settings and [create a token with appropriate permissions.](https://ginnyfahs.medium.com/github-error-authentication-failed-from-command-line-3a545bfd0ca8)
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/Marco-UNIBE-BME/MIA_LAB_24.git
After running this command, Git will prompt you for your GitHub username and your personal access token instead of your password.

Git Setup
Before committing any changes, configure your Git name and email for proper attribution:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
Staging and Committing Changes
```
### Git operates in stages: files can be modified, staged, and then committed.

Check the status of your files:
```bash
git status
```
Stage changes:
```bash
git add file-name
```
Or to stage all modified files:
```bash
git add .
```
Commit your changes:
```bash
git commit -m "Describe what you've done"
```
### Branch Handling
Branching allows you to work on features independently.

Check your current branch:
```bash
git branch
```
Create a new branch:
```bash
git checkout -b my-feature-branch
```
Switch between branches:
```bash
git checkout branch-name
```
Push your branch to GitHub:
```bash
git push -u origin my-feature-branch
```
Merge your branch into main once your work is complete:
```bash
git checkout main
git merge my-feature-branch
```
Delete the branch after merging:
```bash
git branch -d my-feature-branch
```
### Fetching and Pulling Changes
To keep your local repository up-to-date with the remote one:

Fetch changes:
```bash
git fetch
```
Pull changes (fetch and merge):
```bash
git pull
```
Viewing Commit History
To see the commit history:

```bash
git log
```
Or for a compact view:

```bash
git log --oneline
```
### Handling Merge Conflicts
Conflicts can occur when merging branches or pulling changes from the remote.

Identify conflicts with:
```bash
git status
```
Resolve conflicts manually in the conflicting files, then mark them as resolved:
```bash
git add resolved-file
```
Complete the merge:
```bash
git commit
```
### Undoing Changes
If you make a mistake, Git allows you to undo changes.

Unstage files:
```bash
git restore --staged file-name
```
Discard changes in a file:
```bash
git restore file-name
```
Undo a commit but keep changes:
```bash
git reset --soft HEAD~1
```
Or undo a commit and discard the changes:
```bash
git reset --hard HEAD~1
```
### Remote Repository Basics
To interact with the remote repository:

View remotes:
```bash
git remote -v
```
Add a remote:
```bash
git remote add origin https://github.com/username/repository.git
```
Push changes to remote:
```bash
git push origin branch-name
```
By following these steps, you can efficiently manage your branches, handle changes, and collaborate on this repository. For more advanced topics, explore the [official Git documentation.](https://git-scm.com/doc)

September 2024
