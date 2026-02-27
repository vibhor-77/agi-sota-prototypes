import os
import sys

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.arc.env import Grid
from domains.arc.dsl import rotate90, mirror_x, replace_color

def main():
    print("\n==================================")
    print(" 💠 ARC Spatial Grammar Example 💠")
    print("==================================\n")

    print("[1] The AGI views the world as a structured Grid.\n")
    # We present a 3x3 grid
    initial_state = Grid([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2]
    ])

    print("--- Original Grid ---")
    print(initial_state.arr)

    print("\n[2] It applies spatial abstraction primitives (the Action Grammar) to manipulate the world.")
    
    # 1. Rotate the grid 90 degrees
    rotated = rotate90(initial_state)
    print("\n--- After 'rotate90' ---")
    print(rotated.arr)

    # 2. Mirror it across the X-axis
    mirrored = mirror_x(rotated)
    print("\n--- After 'mirror_x' ---")
    print(mirrored.arr)

    # 3. Replace color 1 with color 4 (Yellow)
    final = replace_color(mirrored, 1, 4)
    print("\n--- After 'replace_color(1 -> 4)' ---")
    print(final.arr)

    print("\n🚀 INSIGHT:")
    print("Instead of a neural network guessing pixels, the Beam Search explores")
    print("infinite mathematical combinations of these exact functional operations")
    print("to discover the 'program' that maps Input A exactly to Output B!")
    print("==================================\n")

if __name__ == "__main__":
    main()
