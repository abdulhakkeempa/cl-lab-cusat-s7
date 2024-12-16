class EditDistance:
    def __init__(self):
        self.INSERT_COST = 1
        self.DELETE_COST = 1
        self.REPLACE_COST = 2

    def minimum_edit_distance(self, source: str, target: str):
        """
        Calculate minimum edit distance between source and target strings.
        Returns the distance and the operations needed.
        """
        m, n = len(source), len(target)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        operations = [[None] * (n + 1) for _ in range(m + 1)]


        for i in range(m + 1):
            dp[i][0] = i * self.DELETE_COST
            print(dp)
            if i > 0:
                operations[i][0] = ('DELETE', i-1, 0)

        for j in range(n + 1):
            dp[0][j] = j * self.INSERT_COST
            if j > 0:
                operations[0][j] = ('INSERT', 0, j-1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if source[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    operations[i][j] = ('COPY', i-1, j-1)
                else:
                    replace = dp[i-1][j-1] + self.REPLACE_COST
                    delete = dp[i-1][j] + self.DELETE_COST
                    insert = dp[i][j-1] + self.INSERT_COST

                    min_cost = min(replace, delete, insert)
                    dp[i][j] = min_cost

                    if min_cost == replace:
                        operations[i][j] = ('REPLACE', i-1, j-1)
                    elif min_cost == delete:
                        operations[i][j] = ('DELETE', i-1, j)
                    else:
                        operations[i][j] = ('INSERT', i, j-1)

        print(dp)

        edit_sequence = []
        i, j = m, n

        while i > 0 or j > 0:
            operation, prev_i, prev_j = operations[i][j]

            if operation == 'COPY':
                edit_sequence.append(f"Copy '{source[i-1]}'")
            elif operation == 'REPLACE':
                edit_sequence.append(f"Replace '{source[i-1]}' with '{target[j-1]}'")
            elif operation == 'DELETE':
                edit_sequence.append(f"Delete '{source[i-1]}'")
            else:
                edit_sequence.append(f"Insert '{target[j-1]}'")

            i, j = prev_i, prev_j

        edit_sequence.reverse()

        return dp[m][n], edit_sequence

    def print_detailed_output(self, source: str, target: str):
        """Print detailed output including the edit distance and operations"""
        distance, operations = self.minimum_edit_distance(source, target)

        print(f"Source string: {source}")
        print(f"Target string: {target}")
        print(f"Minimum Edit Distance: {distance}")
        print("\nEdit Operations:")
        for i, op in enumerate(operations, 1):
            print(f"{i}. {op}")


def test_edit_distance():
    ed = EditDistance()
    ed.print_detailed_output("cat", "cut")

    # # Test cases
    # test_cases = [
    #     ("kitten", "sitting"),
    #     ("sunday", "saturday"),
    #     ("intention", "execution"),
    #     ("cat", "cut"),
    #     ("", "hello"),
    #     ("algorithm", "logarithm"),
    #     ("hello", "hello"),
    # ]

    # print("Testing Minimum Edit Distance Algorithm")
    # print("=" * 50)

    # for source, target in test_cases:
    #     print("\nTest Case:")
    #     print("-" * 50)
    #     ed.print_detailed_output(source, target)
    #     print("-" * 50)

if __name__ == "__main__":
    test_edit_distance()
