"""
DPLL SAT Solver - Core Implementation
"""

import copy
from typing import List, Tuple, Dict, Optional


class DPLLSolver:
    """
    A simple DPLL (Davis-Putnam-Logemann-Loveland) SAT solver.

    This solver can determine if a boolean formula in CNF (Conjunctive Normal Form)
    is satisfiable and find a satisfying assignment if one exists.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        # Statistics tracking
        self.stats = {
            "decisions": 0,
            "unit_propagations": 0,
            "backtracks": 0,
            "conflicts": 0,
        }

    def solve(self, cnf_formula: List[List[int]]) -> Tuple[bool, Dict[int, bool]]:
        """
        Main entry point to solve a CNF formula.

        Args:
            cnf_formula: List of clauses, where each clause is a list of literals
                        Example: [[1, -2], [2, 3], [-3]] represents:
                        (x1 OR NOT x2) AND (x2 OR x3) AND (NOT x3)

        Returns:
            Tuple (is_satisfiable, assignment)
            - is_satisfiable: Boolean indicating if formula is satisfiable
            - assignment: Dictionary mapping variables to True/False (if satisfiable)
        """
        if self.verbose:
            print(f"Starting DPLL solver with formula: {cnf_formula}")

        # Reset statistics
        self.stats = {k: 0 for k in self.stats}

        # Start with empty assignment (no variables assigned yet)
        assignments = {}

        # Call the main DPLL recursive function
        result = self._dpll(cnf_formula, assignments)

        if self.verbose:
            if result[0]:  # If satisfiable
                print(f"Formula is SATISFIABLE!")
                print(f"Solution: {result[1]}")
            else:
                print("Formula is UNSATISFIABLE!")

        return result

    def get_statistics(self) -> Dict[str, int]:
        """Return solving statistics"""
        return self.stats.copy()

    def _dpll(
        self, cnf_formula: List[List[int]], assignments: Dict[int, bool]
    ) -> Tuple[bool, Dict[int, bool]]:
        """
        The main recursive DPLL algorithm.

        Args:
            cnf_formula: Current CNF formula (list of clauses)
            assignments: Current variable assignments (dict)

        Returns:
            Tuple (is_satisfiable, final_assignments)
        """
        if self.verbose:
            print(f"  DPLL called with formula: {cnf_formula}")
            print(f"  Current assignments: {assignments}")

        # BASE CASE 1: If no clauses left, all clauses are satisfied
        if not cnf_formula:
            if self.verbose:
                print("  -> All clauses satisfied! Returning True")
            return (True, assignments)

        # BASE CASE 2: If any clause is empty, we have a contradiction
        for clause in cnf_formula:
            if not clause:  # Empty clause means contradiction
                if self.verbose:
                    print("  -> Found empty clause (contradiction)! Returning False")
                self.stats["conflicts"] += 1
                return (False, {})

        # STEP 1: Unit Propagation
        simplified_formula, new_assignments, conflict = self._unit_propagation(
            cnf_formula, assignments
        )

        if conflict:
            if self.verbose:
                print("  -> Conflict found during unit propagation! Returning False")
            self.stats["conflicts"] += 1
            return (False, {})

        # Update our assignments with the propagated ones
        assignments.update(new_assignments)
        cnf_formula = simplified_formula

        if self.verbose:
            print(f"  After unit propagation - Formula: {cnf_formula}")
            print(f"  After unit propagation - Assignments: {assignments}")

        # Check base cases again after unit propagation
        if not cnf_formula:
            return (True, assignments)

        for clause in cnf_formula:
            if not clause:
                self.stats["conflicts"] += 1
                return (False, {})

        # STEP 2: Choose an unassigned variable to branch on
        variable = self._pick_unassigned_variable(cnf_formula, assignments)

        if variable is None:
            if self.verbose:
                print(
                    "  -> No unassigned variable found, but formula not empty. This is unexpected!"
                )
            return (False, {})

        if self.verbose:
            print(f"  Branching on variable {variable}")

        self.stats["decisions"] += 1

        # STEP 3: Try assigning the variable to True
        if self.verbose:
            print(f"  -> Trying {variable} = True")

        try_true_assignments = copy.deepcopy(assignments)
        try_true_assignments[variable] = True
        try_true_formula = self._simplify_formula(cnf_formula, variable, True)

        result = self._dpll(try_true_formula, try_true_assignments)

        if result[0]:  # If this branch worked
            if self.verbose:
                print(f"  -> Branch {variable} = True succeeded!")
            return result

        # STEP 4: If True didn't work, try False
        if self.verbose:
            print(f"  -> Branch {variable} = True failed, trying {variable} = False")

        try_false_assignments = copy.deepcopy(assignments)
        try_false_assignments[variable] = False
        try_false_formula = self._simplify_formula(cnf_formula, variable, False)

        result = self._dpll(try_false_formula, try_false_assignments)

        if result[0]:
            if self.verbose:
                print(f"  -> Branch {variable} = False succeeded!")
            return result

        # STEP 5: Both branches failed - backtrack
        if self.verbose:
            print(f"  -> Both branches for {variable} failed, backtracking")
        self.stats["backtracks"] += 1
        return (False, {})

    def _unit_propagation(
        self, cnf_formula: List[List[int]], assignments: Dict[int, bool]
    ) -> Tuple[List[List[int]], Dict[int, bool], bool]:
        """
        Performs unit propagation: finds unit clauses (clauses with only one unassigned literal)
        and forces their assignment.

        Args:
            cnf_formula: Current CNF formula
            assignments: Current assignments

        Returns:
            Tuple (new_formula, new_assignments, conflict_found)
        """
        if self.verbose:
            print("    Starting unit propagation...")

        new_assignments = {}
        current_formula = copy.deepcopy(cnf_formula)

        # Keep doing unit propagation until no more unit clauses are found
        changed = True
        while changed:
            changed = False

            for i, clause in enumerate(current_formula):
                if not clause:  # Empty clause = conflict
                    return (current_formula, new_assignments, True)

                # Check how many literals in this clause are unassigned
                unassigned_literals = []
                clause_satisfied = False

                for literal in clause:
                    variable = abs(literal)

                    # Check if this variable is already assigned
                    if variable in assignments or variable in new_assignments:
                        # Get the assigned value
                        assigned_value = assignments.get(
                            variable
                        ) or new_assignments.get(variable)

                        # Check if this literal is satisfied by the assignment
                        if (literal > 0 and assigned_value) or (
                            literal < 0 and not assigned_value
                        ):
                            clause_satisfied = True
                            break
                        # If literal is not satisfied, it's effectively removed from clause
                    else:
                        # This literal is unassigned
                        unassigned_literals.append(literal)

                # If clause is already satisfied, skip it
                if clause_satisfied:
                    continue

                # If no unassigned literals and clause not satisfied = conflict
                if not unassigned_literals:
                    if self.verbose:
                        print(f"    -> Conflict: clause {clause} cannot be satisfied")
                    return (current_formula, new_assignments, True)

                # If exactly one unassigned literal = unit clause
                if len(unassigned_literals) == 1:
                    unit_literal = unassigned_literals[0]
                    variable = abs(unit_literal)
                    value = unit_literal > 0  # True if positive, False if negative

                    if self.verbose:
                        print(
                            f"    -> Unit clause found: {clause}, assigning {variable} = {value}"
                        )

                    # Check for conflict with existing assignments
                    if (
                        variable in new_assignments
                        and new_assignments[variable] != value
                    ):
                        if self.verbose:
                            print(
                                f"    -> Conflict: {variable} already assigned to {new_assignments[variable]}"
                            )
                        return (current_formula, new_assignments, True)

                    new_assignments[variable] = value
                    changed = True
                    self.stats["unit_propagations"] += 1

        # Simplify the formula based on all the new assignments
        for variable, value in new_assignments.items():
            current_formula = self._simplify_formula(current_formula, variable, value)

        if self.verbose:
            print(f"    Unit propagation complete. New assignments: {new_assignments}")
        return (current_formula, new_assignments, False)

    def _simplify_formula(
        self, cnf_formula: List[List[int]], variable: int, value: bool
    ) -> List[List[int]]:
        """
        Simplifies the CNF formula by assigning a value to a variable.

        Args:
            cnf_formula: The CNF formula to simplify
            variable: The variable being assigned (positive integer)
            value: True or False

        Returns:
            Simplified CNF formula
        """
        if self.verbose:
            print(f"    Simplifying formula by setting variable {variable} = {value}")

        new_formula = []

        for clause in cnf_formula:
            new_clause = []
            clause_satisfied = False

            for literal in clause:
                lit_variable = abs(literal)

                if lit_variable == variable:
                    # This literal involves the variable we're assigning
                    if (literal > 0 and value) or (literal < 0 and not value):
                        # This literal is satisfied, so the whole clause is satisfied
                        clause_satisfied = True
                        break
                    # If literal is not satisfied, we simply don't add it to new_clause
                    # (it's effectively removed)
                else:
                    # This literal doesn't involve our variable, keep it
                    new_clause.append(literal)

            # Only add the clause if it's not satisfied
            if not clause_satisfied:
                new_formula.append(new_clause)

        if self.verbose:
            print(f"    Formula after simplification: {new_formula}")
        return new_formula

    def _pick_unassigned_variable(
        self, cnf_formula: List[List[int]], assignments: Dict[int, bool]
    ) -> Optional[int]:
        """
        Picks an unassigned variable to branch on.
        For now, we use a simple heuristic: pick the first unassigned variable we find.

        Args:
            cnf_formula: Current CNF formula
            assignments: Current assignments

        Returns:
            Variable number (positive integer) or None if all assigned
        """
        # Collect all variables that appear in the formula
        variables_in_formula = set()
        for clause in cnf_formula:
            for literal in clause:
                variables_in_formula.add(abs(literal))

        # Find the first unassigned variable
        for variable in sorted(variables_in_formula):
            if variable not in assignments:
                if self.verbose:
                    print(f"    Picked unassigned variable: {variable}")
                return variable

        if self.verbose:
            print("    No unassigned variables found")
        return None
