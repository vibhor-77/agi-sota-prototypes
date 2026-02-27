import os
import sys

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.arc.env import Grid
from domains.arc.dsl import rotate90, flip_y, replace_color

def main():
    print("\n==================================")
    print(" 💠 ARC Spatial Grammar Example 💠")
    print("==================================\n")

    print("[1] The AGI views the world as a structured Grid.\n")
    # We present an asymmetric 3x3 grid
    initial_state = Grid([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print("--- Original Grid ---")
    print(initial_state.arr)

    print("\n[2] It applies spatial abstraction primitives (the Action Grammar) to manipulate the world.")
    
    # 1. Rotate the grid 90 degrees
    rotated = rotate90(initial_state)
    print("\n--- After 'rotate90' ---")
    print(rotated.arr)

    # 2. Flip the grid vertically (up-down)
    mirrored = flip_y(rotated)
    print("\n--- After 'flip_y' ---")
    print(mirrored.arr)

    # 3. Replace color 7 with color 0 (Black)
    final = replace_color(mirrored, 7, 0)
    print("\n--- After 'replace_color(7 -> 0)' ---")
    print(final.arr)

    print("\n🚀 INSIGHT:")
    print("Instead of a neural network guessing pixels, the Beam Search explores")
    print("infinite mathematical combinations of these exact functional operations")
    print("to discover the 'program' that maps Input A exactly to Output B!")
    print("==================================\n")

if __name__ == "__main__":
    main()
