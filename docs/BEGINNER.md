# Beginner guide

This page is for someone who has never opened a terminal. A terminal is a window where you type a line and press Enter. Do the steps in order. The longer manual is [Getting started](GETTING_STARTED.md).

## 1. Download this project

1. Open https://github.com/thcabq352/ltx-video-buddy
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip the download (open the packed file so it becomes a normal folder). The folder is named `ltx-video-buddy-main`. Inside it, open `video_buddy`. That inner folder is the one the later steps use.

If you already use Git (a tool that copies a project and can update it later), you can instead run `git clone https://github.com/thcabq352/ltx-video-buddy.git`. That makes a folder named `ltx-video-buddy`, with `video_buddy` inside it.

## 2. Install Python

Python is the program that runs this project. You need version 3.10 or newer.

1. Download the Windows installer from https://www.python.org/downloads/
2. On the first installer screen, tick **Add python.exe to PATH**. PATH is the list Windows searches when you type a program's name.
3. Finish the install, then close the installer.

On a Mac or Linux computer, install Python 3.10 or newer the usual way for that computer. If it is missing, the installer in step 5 prints `FAIL  Python 3.10+ not found.` and names the install command.

## 3. Open a terminal

On Windows: click **Start**, type `cmd`, and press Enter. The window that opens is the terminal.

On a Mac, open the app named **Terminal**. On Linux, open **Terminal** the same way.

## 4. Go into the video_buddy folder

The terminal starts in your home folder. `cd` means "change directory" (move into a folder). Use your own user name in place of `You`:

```
cd C:\Users\You\Downloads\ltx-video-buddy-main\video_buddy
```

Tip: type `cd` and a space, drag the `video_buddy` folder from File Explorer into the window, then press Enter.

## 5. Run the installer

In that same window, type this and press Enter:

```
install.bat --automatic-install --yes
```

`--automatic-install` means the full setup (the video program, the video model, and a local helper). `--yes` means go ahead after a check. Leave the window open. The download is large and can take a long time.

On a Mac or Linux, from the same `video_buddy` folder, run this instead:

```
./install.sh --automatic-install --yes
```

## 6. What the pre-flight check means

A pre-flight check is a look-over that runs before anything is downloaded or installed. The first line is:

`AUTOMATIC INSTALL — pre-flight check (nothing has been changed yet)`

Each check is labeled `[PASS]`, `[FAIL]`, or `[INFO]`. This check does not print OK or WARN.

- `[PASS]` means that check is fine.
- `[FAIL]` means stop. The lines under the name say what is wrong. Then you see `STOPPED at '...'`. Fix the item above, then run automatic install again. You also see `Nothing was downloaded or installed.`
- `[INFO]` is a note. It never stops the install. The graphics-card check and the llama.cpp check (the local helper program) are INFO.

The five checks, in order: Disk space, Hugging Face access (the website the model files come from), GPU / VRAM (your graphics card and its memory), llama.cpp, and ComfyUI folder (where the video program will live).

When nothing blocking failed, the last line is `Pre-flight passed.` and the install continues.

## 7. If you see FAIL

Read the lines under `[FAIL]`, or a line that starts with `FAIL`. Do what they say, then run the same install command again. Running it again is safe. Work that already finished is skipped.

Common fixes:

- **Not enough disk space.** The Disk space check says `NOT enough` and how much to free. Empty the Recycle Bin (Trash on a Mac), delete old downloads, or follow its line about a bigger drive. Then run the command again.
- **Python was not found.** The line is `FAIL  Python 3.10+ not found.` Install Python again, tick **Add python.exe to PATH**, close the terminal, open a new one, and start again at step 4.
- **The ComfyUI folder cannot be used.** The message says the folder already holds other files, cannot be written to, or is set to a video program you manage yourself. Follow that message, then run the command again.
- **A saved sign-in token does not work.** The usual files need no token. If one is set and it is bad, the Hugging Face check says so and lists the steps: sign in, create a Read token, save it in a file, and run again.

If a required step fails after the pre-flight, the end says `Not ready for generate yet. Still needs you:` and `Install FAILED:`. Fix the named item and run the same command again.

At the end, a short read-only check uses different words: `OK` (that piece is ready), `WARN` (only for missing local helper files, next to `llm-models`, with `WARNING: local LLM files are missing.`), `INFO` (a note), or `NEED` (something required is still missing). `WARN` is not a pre-flight label.

Still stuck? Open an issue and paste the FAIL lines: https://github.com/thcabq352/ltx-video-buddy/issues

## 8. When it says Ready

Wait until you see `Install OK:` and then:

`Ready for generate (LTX 2.3). Next:`

Stay in the `video_buddy` folder. Run the first command and wait until it finishes. Then run the second.

```
python -m master_agent comfy start
python -m master_agent comfy run --mode generate --variant base --prompt "a test shot"
```

The first starts ComfyUI, the program on your computer that makes the video. The second asks it to make a short test clip from the words inside the quotes.
