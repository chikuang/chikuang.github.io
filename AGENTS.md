# Website publishing instructions

## New-publication updates

Standing user instruction (2026-10-27): when the user provides a new paper or
requests a publication update, complete the website Publication entry, related
student introductions, CV source and PDF, and GitHub profile CV-link verification
as one task. These routine edits, rendering, committing, and publishing are
already authorized; do not ask for separate approval. This is a workflow for
requested updates, not a request for unattended monitoring or a schedule.

- Fetch the latest source and publishing branches before editing. Preserve
  unrelated work and stage only the task's changes.
- Verify the title, author order, year/status, arXiv/DOI, and code links against
  the paper's authoritative sources. Avoid duplicate entries and preserve
  existing student markers and award attribution.
- Update publications in `website/config.toml` using the existing format and
  sorting. Add the paper to each relevant student's Group introduction using
  the next available P-number; keep existing paper and software links intact.
- Update the latest canonical CV source and its editorial date. The current
  source is Overleaf project `5b8fdcb00cf8821d571632b4`, `Long/Long_CV.tex` and
  `Long/publications.bib`, synced locally under
  `~/Library/CloudStorage/Dropbox/Apps/Overleaf/CV-Chi-Kuang Yeh/Long/`.
  Check for an existing entry or a completed update before adding it again.
  Use a newer confirmed canonical source if the user has changed it.
- Compile the current CV, or retrieve its verified freshly compiled Overleaf
  PDF. Inspect the PDF pages, new citation, date, and clickable paper/code
  links. Replace both `website/static/CV.pdf` and `chikuang.github.io/CV.pdf`
  with identical verified bytes; preserve dated CV archives.
- Check the live `chikuang/chikuang` profile README's Curriculum Vitae link.
  Prefer the stable `https://chikuang.github.io/CV.pdf` URL. Fix an outdated
  target if needed; if already correct, verify that it serves the latest PDF
  without making a needless profile edit.
- Render Hugo into a temporary directory and synchronize the affected generated
  pages. Never use `--cleanDestinationDir` on the publishing checkout. Preserve
  independently maintained course material and other unrelated site content.
- Commit and push the website source and publishing changes, plus profile
  changes if any. Keep affected primary local checkouts synchronized with
  GitHub. Verify deployed Publication/Group entries and download the public
  CV URL to compare its bytes with the verified PDF. Report completion only
  after deployment and link verification, or identify the exact unfinished step.

## Teaching publication

These instructions apply to teaching-related work in this publishing repository.
Edit canonical course sources first, render them, then copy only the affected
course output here. Preserve other courses, website content, and CVs.

| Published directory | Primary course source | Branch |
| --- | --- | --- |
| teaching/2026F_stat8670/ | ~/Documents/GitHub/teaching/Teaching-Stat8670 | 2026F |
| teaching/2026F_stat8561/ | ~/Documents/GitHub/teaching/Teaching-Stat8561 | master |
| teaching/genai/ | ~/Desktop/Teaching-GenAI | main |
| teaching/statistical-learning/ | ~/Desktop/Teaching-SML-work/Teaching-SML | main |

Follow each source repository's AGENTS.md and documented build. The statistics
courses publish `_book/`; GenAI and SML publish `_build/html/`.

### Teaching publication and primary-checkout synchronization

Standing user instruction (2026-10-22): every teaching-material update must finish
with rendering, committing, pushing, and verification that the primary local
repositories match GitHub. Rendering and uploading are already authorized.

- Before editing, fetch the affected primary source and publishing repositories
  and start from the latest current-course branch. Prefer the primary checkouts;
  preserve existing local edits and avoid working from stale copies.
- Render the affected teaching material to HTML and PDF with the documented
  project build. Execute changed examples and check affected figures, numerical
  outputs, links, and PDF layout. Keep both formats consistent.
- Commit and push the source plus tracked generated artifacts. Publish the
  rendered course output to its directory in `chikuang/chikuang.github.io`, then
  commit and push the publishing repository. For teaching-listing changes,
  render and push the affected `chikuang/website` source and published pages too.
  Preserve other courses and website content; do not clean the whole website.
- After pushing from any temporary clone or worktree, fetch and fast-forward the
  corresponding primary local source and publishing checkouts. Updating only a
  temporary copy is not completion. Synchronize other clean copies used during
  the task as well. Reconcile local edits/divergence without losing work.
- Before reporting completion, fetch again, verify the actual GitHub branch SHA
  equals local HEAD, and require `git rev-list --left-right --count
  HEAD...@{upstream}` to report `0 0` in every affected primary checkout. Check
  `git status --short --branch` for uncommitted or unpushed task changes. Preserve
  and explicitly report unrelated pre-existing work.
- Verify the deployed course page and PDF. Return versioned/refreshable links,
  the relevant commits, and the primary-checkout synchronization result. Repair
  failures or report the exact unfinished step; do not claim completion early.

The primary website source is `~/Documents/GitHub/personal_web/website` (`master`).
The primary publishing checkout is
`~/Documents/GitHub/personal_web/chikuang.github.io` (`master`).
