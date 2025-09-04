import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
            #self.domains是字典，key是var，value是words的copy
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")#如果letters[i][j]为空，则打印空格，否则打印字母
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("D:\\Users\\UNO\\Desktop\\cs\\HAVARD_cs50\\project\\hw3_optimization\\crossword\\assets\\fonts\\OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            for value in list(self.domains[var]):
                if len(value) != var.length:
                    self.domains[var].remove(value)
        

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        
        if overlap is None:
            return False
            
        i, j = overlap
        
        # 检查x的每个值是否有对应的y值
        for x_word in list(self.domains[x]):
            found = False
            for y_word in self.domains[y]:
                if x_word[i] == y_word[j]:
                    found = True
                    break
                    
            if not found:
                self.domains[x].remove(x_word)
                revised = True
                
        return revised


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = self.get_all_arcs()  # 假设你有一个方法可以获取所有弧的列表

        while arcs:
            x, y = arcs.pop(0)  # 弹出弧列表中的第一对变量
            if self.revise(x, y):  # 假设你有一个方法可以处理弧一致性
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x):  
                    if z != y:
                        arcs.append((z, x))  # 如果x的domian被修订，将其所有弧添加到队列中，除了(y,x)

        return True


    def get_all_arcs(self):
        """
        Generate a list of all arcs in the problem.
        """
        arcs = []
        for x in self.crossword.variables:  
            for y in self.crossword.neighbors(x):  
                arcs.append((x, y))
        return arcs
        

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return len(assignment) == len(self.crossword.variables)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # 检查所有值是否不同
        values = list(assignment.values())
        if len(values) != len(set(values)):
            return False
            
        # 检查所有值长度是否正确
        for var, value in assignment.items():
            if len(value) != var.length:
                return False

         # 检查重叠部分是否一致
        for var1 in assignment:
            for var2 in assignment:
                if var1 != var2:
                    overlap = self.crossword.overlaps[var1, var2]
                    if overlap is not None:
                        i, j = overlap
                        if assignment[var1][i] != assignment[var2][j]:
                            return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        value_constraints = {}
        neighbors = self.crossword.neighbors(var)
        
        for value in self.domains[var]:
            # 跳过已分配的邻居
            unassigned_neighbors = [n for n in neighbors if n not in assignment]
            
            # 计算这个值会排除多少邻居的可能值
            ruled_out = 0
            for neighbor in unassigned_neighbors:
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is not None:
                    i, j = overlap
                    for neighbor_value in self.domains[neighbor]:
                        if value[i] != neighbor_value[j]:
                            ruled_out += 1
                
            value_constraints[value] = ruled_out
        
        # 按约束程度排序（最少约束的值优先）
        return sorted(self.domains[var], key=lambda x: value_constraints[x])


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned = [var for var in self.crossword.variables if var not in assignment]
         # 按剩余值数量排序
        unassigned.sort(key=lambda var: len(self.domains[var]))
        
        # 如果有多个变量有相同的最小剩余值数量，按度（邻居数量）排序
        min_remaining = len(self.domains[unassigned[0]])
        candidates = [var for var in unassigned if len(self.domains[var]) == min_remaining]
        
        if len(candidates) > 1:
            candidates.sort(key=lambda var: len(self.crossword.neighbors(var)), reverse=True)
            
        return candidates[0]
    
    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
            
        # 选择一个未赋值的变量
        var = self.select_unassigned_variable(assignment)
        
        # 尝试变量的每个可能值
        for value in self.order_domain_values(var, assignment):
            # 创建新的赋值
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            # 检查一致性
            if self.consistent(new_assignment):
                # 向前检查（可选步骤）
                domains_backup = {v: self.domains[v].copy() for v in self.domains}
                self.domains[var] = {value}
                
                # 递归搜索
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
                    
                # 恢复域（回溯）
                self.domains = domains_backup
                
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
