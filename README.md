<div align="center">
    <h1><strong>TRM</strong>: Tiny Reasoning Model</h1>
</div>

This is a project aimed to create a lightweight TRM that can solve sudoku puzzles using logical reasoning techniques. The model is designed to be efficient and easy to use, making it suitable for educational purposes and small-scale applications.

To learn more about how this project is setup and the model works, see the [documentation](docs/documentation.md).

## Features

- **Lightweight**: The model is designed to be small and efficient, making it easy to run on various devices.
- **Logical Reasoning**: Utilizes logical reasoning techniques to solve sudoku puzzles.
- **Easy to Use**: Simple interface for users to input puzzles and receive solutions.
- **Educational**: A great tool for learning about logical reasoning and problem-solving techniques.

## Installation

To install the Tiny Reasoning Model, clone the repository and install the required dependencies:

```bash
git clone https://github.com/s4nj1th/tiny-reasoning-model.git
cd tiny-reasoning-model
pip install -r requirements.txt
```

## Usage

0. Prepare your environment by ensuring you have Python 3.7+ and the required libraries installed.

   - Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure you have a compatible GPU if you plan to train the model.
   - Store the dataset of sudoku puzzles in the `data/` directory (as `sudoku.csv`).

1. Run the `tiny-reasoning-model.ipynb` notebook.

   - This trains and tests the TRM on sudoku puzzles.
   - The models are saved at `checkpoints/`.

2. Run the `main.py` script to input a sudoku puzzle and get the solution.

   - Input puzzles can be provided in a text file format.
   - Example command:

     ```bash
     python main.py --input puzzle.txt --output solution.txt
     ```

     > **Note**: The input puzzle should be in a 9x9 grid format, with empty cells represented by zeros.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
