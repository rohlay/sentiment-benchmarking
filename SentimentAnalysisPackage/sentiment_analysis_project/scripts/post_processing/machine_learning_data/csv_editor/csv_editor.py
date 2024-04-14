#from csv_editor_remove import apply_modifications_to_csvs
#from csv_editor_add import merge_data_from_two_folders
from sentiment_analysis_project.scripts.csv_files_editor.csv_editor_remove import apply_modifications_to_csvs
from sentiment_analysis_project.scripts.csv_files_editor.csv_editor_add import merge_data_from_two_folders

def main():
    while True:
        choice = input("Choose editor: add, remove or exit: ").lower()

        if choice == "exit":
            print("Exiting the program.")
            break
        elif choice == "add":
            ID1 = input("Enter ID1: ")
            ID2 = input("Enter ID2: ")
            ID3 = input("Enter ID3: ")

            models = input("Enter models to add (comma-separated): ").split(',')
            datasets = input("Enter datasets to add (comma-separated): ").split(',')

            merge_data_from_two_folders(ID1, ID2, ID3, datasets, models)

        elif choice == "remove":
            IDENTIFIER = input("Enter IDENTIFIER: ")
            CLASS_OR_REG = input("Enter either 'classification' or 'regression': ")

            models = input("Enter models to remove (comma-separated): ").split(',')
            datasets = input("Enter datasets to remove (comma-separated): ").split(',')

            apply_modifications_to_csvs(IDENTIFIER, CLASS_OR_REG, datasets, models)

        else:
            print("Invalid choice. Please enter 'add', 'remove', or 'exit'.")

if __name__ == "__main__":
    main()
