import os
import shutil
import math
import pandas as pd


def create_mini_dataset(
    csv_path: str,
    moods: list,
    file_count: int = 50,
    export_selection: bool = False,
    source_audio_dir: str = "./files",
):
    max_per_label = math.ceil(file_count / len(moods))

    df = pd.read_csv(csv_path)

    selected_rows = []
    label_counts = {mood: 0 for mood in moods}

    for _, row in df.iterrows():
        if len(selected_rows) >= file_count:
            break

        raw_tags = str(row["TAGS"]).split("+")
        cleaned_tags = [tag.replace("mood/theme---", "") for tag in raw_tags]

        # LOGIC CHECK: Are ALL the track's tags present in our target 'moods' list?
        # If even one tag is not in our list (e.g., tag "D" when list is [A, B, C]), it is discarded.
        if all(tag in moods for tag in cleaned_tags):
            first_label = cleaned_tags[0]

            if label_counts[first_label] < max_per_label:
                new_row = row.copy()
                new_row["TAGS"] = "+".join(cleaned_tags)
                new_row["PATH"] = f"00/{new_row["PATH"].split("/")[-1]}"

                selected_rows.append(new_row)
                label_counts[first_label] += 1

    mini_df = pd.DataFrame(selected_rows)

    # Debug
    print(f">>> Mini dataset created with length of {len(mini_df)} tracks")
    print("Label distribution:")
    for mood, count in label_counts.items():
        print(f"  - {mood}: {count}")
    print("\nDF preview:")
    print(mini_df.to_string(index=False))
    print("-" * 50)

    # Export
    if export_selection:
        dest_dir = os.path.join(".", "tempdemo", "raw", "00")
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\nExporting files to: {dest_dir}")

        success_count = 0
        for _, row in mini_df.iterrows():
            src_path = os.path.join(source_audio_dir, row["PATH"])
            dest_path = os.path.join(dest_dir, os.path.basename(row["PATH"]))

            try:
                shutil.copy2(src_path, dest_path)
                success_count += 1
            except FileNotFoundError:
                print(f">>> [Warning] Audio file not found, skipping copy: {src_path}")

        print(f"Successfully copied {success_count} out of {len(mini_df)} files.")
    else:
        print("\nexport_selection is False. No directories created or files copied.")

    return mini_df


if __name__ == "__main__":
    moods_list = ["relaxing", "melodic", "emotional"]

    # CHANGE THESE
    csv_file = "../prep/autotagging_moodtheme-train_clean.csv"
    split = "train"

    try:
        final_df = create_mini_dataset(
            csv_path=csv_file,
            moods=moods_list,
            file_count=60,
            export_selection=True,
            source_audio_dir="/path/to/rawdata",
        )

        os.makedirs("./tempdemo/csv/", exist_ok=True)
        final_df.to_csv(f"./tempdemo/csv/mini_dataset_{split}.csv", index=False)

    except FileNotFoundError:
        print("Please ensure the CSV file path is correct to run the test.")
