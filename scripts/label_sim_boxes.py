"""Label sim box images by profile using a simple GUI.

Shows each unlabeled image and lets you click a profile button.
Progress is saved after each label so you can quit and resume anytime.

Usage:
    python scripts/label_sim_boxes.py                # label unlabeled only
    python scripts/label_sim_boxes.py --review       # review all, correct mistakes
    python scripts/label_sim_boxes.py --review-only  # review labeled only
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

DEFAULT_IMAGES_DIR = Path("training/dataset/v1/sim/images")
PROFILES = ["normal", "fragile", "heavy"]
DISPLAY_SIZE = (640, 480)


def load_labels(labels_path: Path) -> dict[str, str]:
    """Load existing labels from disk."""
    if labels_path.exists():
        with open(labels_path) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict[str, str], labels_path: Path) -> None:
    """Save labels to disk."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)


def collect_unlabeled(images_dir: Path, labels: dict[str, str]) -> list[Path]:
    """Collect all PNG images not yet labeled."""
    all_images: list[Path] = []
    for split in ["train", "eval"]:
        split_dir = images_dir / split
        if split_dir.is_dir():
            all_images.extend(sorted(split_dir.glob("*.png")))
    return [p for p in all_images if p.name not in labels]


PROFILE_COLORS = {"normal": "#4CAF50", "fragile": "#FF9800", "heavy": "#2196F3"}
PROFILE_KEYS = {"normal": "1", "fragile": "2", "heavy": "3"}


class LabelApp:
    def __init__(
        self,
        images: list[Path],
        labels: dict[str, str],
        labels_path: Path,
        total_count: int,
    ) -> None:
        self.images = images
        self.labels = labels
        self.labels_path = labels_path
        self.total_count = total_count
        self.index = 0

        self.root = tk.Tk()
        self.root.title("Sim Box Labeler")
        self.root.configure(bg="#1e1e1e")

        # Status bar
        self.status_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("monospace", 14),
            bg="#1e1e1e",
            fg="#cccccc",
        ).pack(pady=(10, 5))

        # Current label indicator
        self.current_label_var = tk.StringVar()
        self.current_label_widget = tk.Label(
            self.root,
            textvariable=self.current_label_var,
            font=("monospace", 14, "bold"),
            bg="#1e1e1e",
            fg="#888888",
        )
        self.current_label_widget.pack(pady=(0, 5))

        # Image display
        self.canvas = tk.Label(self.root, bg="#2d2d2d")
        self.canvas.pack(padx=10, pady=5)

        # Filename
        self.filename_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.filename_var,
            font=("monospace", 11),
            bg="#1e1e1e",
            fg="#888888",
        ).pack(pady=(0, 5))

        # Profile buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=5)

        self.profile_btns: dict[str, tk.Button] = {}
        for profile in PROFILES:
            key = PROFILE_KEYS[profile]
            btn = tk.Button(
                btn_frame,
                text=f"  {profile.upper()} ({key})  ",
                font=("monospace", 16, "bold"),
                bg=PROFILE_COLORS[profile],
                fg="white",
                activebackground=PROFILE_COLORS[profile],
                activeforeground="white",
                relief="flat",
                padx=20,
                pady=10,
                command=lambda p=profile: self.assign(p),
            )
            btn.pack(side=tk.LEFT, padx=8)
            self.profile_btns[profile] = btn
            self.root.bind(key, lambda e, p=profile: self.assign(p))

        # Navigation frame
        nav_frame = tk.Frame(self.root, bg="#1e1e1e")
        nav_frame.pack(pady=(5, 10))

        tk.Button(
            nav_frame,
            text="  BACK (A)  ",
            font=("monospace", 14, "bold"),
            bg="#555555",
            fg="white",
            relief="flat",
            padx=15,
            pady=8,
            command=self.go_back,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            nav_frame,
            text="  KEEP (Enter)  ",
            font=("monospace", 14, "bold"),
            bg="#555555",
            fg="white",
            relief="flat",
            padx=15,
            pady=8,
            command=self.keep,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            nav_frame,
            text="  SKIP (S)  ",
            font=("monospace", 14, "bold"),
            bg="#555555",
            fg="white",
            relief="flat",
            padx=15,
            pady=8,
            command=self.skip,
        ).pack(side=tk.LEFT, padx=8)

        # Key bindings
        self.root.bind("a", lambda e: self.go_back())
        self.root.bind("<Left>", lambda e: self.go_back())
        self.root.bind("<Return>", lambda e: self.keep())
        self.root.bind("s", lambda e: self.skip())
        self.root.bind("<Right>", lambda e: self.skip())
        self.root.bind("<Control-z>", lambda e: self.undo())

        self.show_current()

    def show_current(self) -> None:
        """Display the current image."""
        if self.index >= len(self.images):
            labeled_count = sum(1 for img in self.images if img.name in self.labels)
            self.status_var.set(f"Done! {labeled_count}/{len(self.images)} labeled.")
            self.current_label_var.set("")
            self.filename_var.set("")
            self.canvas.configure(image="")
            return

        path = self.images[self.index]
        remaining = len(self.images) - self.index
        self.status_var.set(f"[{self.index + 1}/{len(self.images)}]  {remaining} remaining")
        self.filename_var.set(f"{path.parent.name}/{path.name}")

        # Show current label
        current = self.labels.get(path.name)
        if current:
            color = PROFILE_COLORS.get(current, "#888888")
            self.current_label_var.set(f"current: {current.upper()}")
            self.current_label_widget.configure(fg=color)
        else:
            self.current_label_var.set("current: unlabeled")
            self.current_label_widget.configure(fg="#888888")

        # Highlight active button
        for profile, btn in self.profile_btns.items():
            if profile == current:
                btn.configure(relief="sunken", borderwidth=3)
            else:
                btn.configure(relief="flat", borderwidth=1)

        img = Image.open(path)
        img.thumbnail(DISPLAY_SIZE, Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self._photo)

    def assign(self, profile: str) -> None:
        """Label the current image and advance."""
        if self.index >= len(self.images):
            return
        path = self.images[self.index]
        self.labels[path.name] = profile
        save_labels(self.labels, self.labels_path)
        self.index += 1
        self.show_current()

    def keep(self) -> None:
        """Keep the current label and advance."""
        if self.index >= len(self.images):
            return
        self.index += 1
        self.show_current()

    def skip(self) -> None:
        """Skip the current image without labeling."""
        if self.index >= len(self.images):
            return
        self.index += 1
        self.show_current()

    def go_back(self) -> None:
        """Go back to the previous image."""
        if self.index <= 0:
            return
        self.index -= 1
        self.show_current()

    def undo(self) -> None:
        """Go back and remove the previous label."""
        if self.index <= 0:
            return
        self.index -= 1
        path = self.images[self.index]
        self.labels.pop(path.name, None)
        save_labels(self.labels, self.labels_path)
        self.show_current()

    def run(self) -> None:
        self.root.mainloop()


def collect_all(images_dir: Path) -> list[Path]:
    """Collect all PNG images across splits."""
    images: list[Path] = []
    for split in ["train", "eval"]:
        split_dir = images_dir / split
        if split_dir.is_dir():
            images.extend(sorted(split_dir.glob("*.png")))
    return images


def print_summary(labels: dict[str, str], total: int) -> None:
    from collections import Counter

    counts = Counter(labels.values())
    print(f"\nLabeled {len(labels)}/{total} images:")
    for profile in PROFILES:
        print(f"  {profile}: {counts.get(profile, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label sim box images by profile")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Labels JSON file (default: <images-dir>/labels.json)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--review",
        action="store_true",
        help="Review ALL images (labeled + unlabeled), correct mistakes",
    )
    mode.add_argument(
        "--review-only",
        action="store_true",
        help="Review only already-labeled images",
    )
    args = parser.parse_args()

    labels_path = args.labels or (args.images_dir / "labels.json")
    labels = load_labels(labels_path)
    all_images = collect_all(args.images_dir)
    total = len(all_images)

    if args.review:
        show_images = all_images
        mode_str = "review (all)"
    elif args.review_only:
        show_images = [p for p in all_images if p.name in labels]
        mode_str = "review (labeled only)"
    else:
        show_images = collect_unlabeled(args.images_dir, labels)
        mode_str = "label (unlabeled only)"

    labeled_count = total - len(collect_unlabeled(args.images_dir, labels))
    print(f"Mode: {mode_str}")
    print(f"Total images: {total}")
    print(f"Already labeled: {labeled_count}")
    print(f"Showing: {len(show_images)}")
    print(f"Labels file: {labels_path}")
    print()
    print("Keys: 1=normal  2=fragile  3=heavy  Enter=keep  A/Left=back  S/Right=skip  Ctrl+Z=undo")
    print()

    if not show_images:
        print("Nothing to show!")
        print_summary(labels, total)
        return

    app = LabelApp(show_images, labels, labels_path, total)
    app.run()

    print_summary(labels, total)


if __name__ == "__main__":
    main()
