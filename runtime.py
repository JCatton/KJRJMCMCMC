# runtime.py

"""
This file defines the RuntimePage class, which provides the interface for loading and running MCMC simulations.

It allows the user to load an existing MCMC object, create a new one with specified parameters, and run the Metropolis-Hastings algorithm.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QSpinBox, QFormLayout, QTextEdit, QMessageBox, QGridLayout, QComboBox
)
from PySide6.QtCore import Slot, Qt
from MCMC.mcmc import MCMC
from pathlib import Path
import numpy as np
import threading

class RuntimePage(QWidget):
    """
    The Runtime page, providing controls to load and run MCMC simulations.
    """

    def __init__(self):
        super().__init__()

        self.mcmc = None  # Placeholder for the MCMC object
        self.raw_data_path = None
        self.input_fields = {}  # Nested dictionary to store input fields
        self.parameter_names = [r"\eta", "a", "P", "e", "i", r"\omega", r"\Omega", r"\phi", "mass"]

        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()

        # Load existing MCMC
        load_layout = QHBoxLayout()
        self.load_label = QLabel("Load MCMC Directory:")
        self.load_button = QPushButton("Browse")
        self.load_button.clicked.connect(self.load_mcmc)
        load_layout.addWidget(self.load_label)
        load_layout.addWidget(self.load_button)
        left_layout.addLayout(load_layout)

        # Raw data selection
        raw_data_layout = QHBoxLayout()
        self.raw_data_button = QPushButton("Select Raw Data File")
        self.raw_data_button.clicked.connect(self.select_raw_data)
        raw_data_layout.addWidget(QLabel("Raw Data:"))
        raw_data_layout.addWidget(self.raw_data_button)
        left_layout.addLayout(raw_data_layout)

        # Number of bodies selection
        bodies_layout = QHBoxLayout()
        self.bodies_label = QLabel("Number of Bodies:")
        self.bodies_spinbox = QSpinBox()
        self.bodies_spinbox.setMinimum(1)
        self.bodies_spinbox.setMaximum(3)
        self.bodies_spinbox.setValue(1)
        self.bodies_spinbox.valueChanged.connect(self.update_body_widgets)
        bodies_layout.addWidget(self.bodies_label)
        bodies_layout.addWidget(self.bodies_spinbox)
        left_layout.addLayout(bodies_layout)

        # Parameter input area
        self.parameter_area = QWidget()
        self.parameter_layout = QVBoxLayout(self.parameter_area)
        left_layout.addWidget(self.parameter_area)

        # Create initial body widgets
        self.body_widgets = []
        self.update_body_widgets()



        self.form_layout = QFormLayout()

        # Add CPU nodes
        self.max_cpu_nodes_input = QSpinBox()
        self.max_cpu_nodes_input.setMinimum(1)
        self.max_cpu_nodes_input.setMaximum(64)
        self.max_cpu_nodes_input.setValue(16)
        self.form_layout.addRow("Max CPU Nodes:", self.max_cpu_nodes_input)

        # Add button for creating new MCMC chain
        self.create_mcmc_button = QPushButton("Create New MCMC")
        self.create_mcmc_button.clicked.connect(self.create_new_mcmc)
        self.form_layout.addRow(self.create_mcmc_button)
        left_layout.addLayout(self.form_layout)
        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        # Run Metropolis-Hastings and IPython console
        rh_layout = QVBoxLayout()
        self.iterations_label = QLabel("Number of New Iterations:")
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setMinimum(1)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_metropolis_hastings)
        rh_layout.addWidget(self.iterations_label)
        rh_layout.addWidget(self.iterations_spinbox)
        rh_layout.addWidget(self.run_button)
        right_layout.addLayout(rh_layout)

        # IPython console placeholder
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setText("IPython console placeholder.")
        right_layout.addWidget(self.console)

        main_layout.addLayout(right_layout)

    @Slot()
    def load_mcmc(self) -> None:
        """Opens a dialog to select an existing MCMC directory and loads the MCMC object."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select MCMC Directory")
        if dir_path:
            try:
                self.mcmc = MCMC.load(Path(dir_path))
                self.console.append(f"Loaded MCMC from {dir_path}")
            except Exception as e:
                self.console.append(f"Error loading MCMC: {e}")

    @Slot()
    def select_raw_data(self) -> None:
        """Opens a dialog to select a raw data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raw Data File", "", "NumPy Files (*.npy)"
        )
        if file_path:
            self.raw_data_path = file_path
            self.console.append(f"Selected raw data file: {file_path}")

    def clear_layout(self, layout):
        """Recursively clears all items from a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
        self.body_widgets = []

    @Slot()
    def update_body_widgets(self) -> None:
        """Updates the body widgets based on the number of bodies selected."""
        # Clear existing widgets and layouts
        self.clear_layout(self.parameter_layout)

        num_bodies = self.bodies_spinbox.value()

        grid_layout = QGridLayout()

        # Add parameter names as headers (columns start from 2 to leave space for row labels)
        for col, param in enumerate(self.parameter_names):
            label = QLabel(f"{param}")
            label.setAlignment(Qt.AlignCenter)
            grid_layout.addWidget(label, 0, col + 2)  # Columns start from 2

        current_row = 1  # Start from row 1 since row 0 is for parameter names

        for body_index in range(num_bodies):
            # Add "Body N" label spanning multiple rows
            body_label = QLabel(f"Body {body_index + 1}")
            body_label.setAlignment(Qt.AlignCenter)
            row_span = 4  # Number of rows per body
            grid_layout.addWidget(body_label, current_row, 0, row_span, 1)  # Column 0

            # Rows: True Values, Initial Values, Proposal Std, Param Bounds
            row_labels = ["True Values", "Initial Values", "Proposal Std", "Param Bounds"]
            for row_offset, row_label in enumerate(row_labels):
                # Add row label
                label = QLabel(row_label)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid_layout.addWidget(label, current_row + row_offset, 1)  # Column 1

                for col, param in enumerate(self.parameter_names):
                    col_index = col + 2  # Start from column 2
                    if row_label == "Param Bounds":
                        # For param bounds, we need two inputs (lower and upper bounds)
                        lower_input = QLineEdit()
                        lower_input.setPlaceholderText("Lower")
                        upper_input = QLineEdit()
                        upper_input.setPlaceholderText("Upper")
                        bounds_layout = QHBoxLayout()
                        bounds_layout.addWidget(lower_input)
                        bounds_layout.addWidget(upper_input)
                        container_widget = QWidget()
                        container_widget.setLayout(bounds_layout)
                        grid_layout.addWidget(container_widget, current_row + row_offset, col_index)
                    else:
                        input_field = QLineEdit()
                        grid_layout.addWidget(input_field, current_row + row_offset, col_index)

            current_row += row_span  # Move to the next set of rows for the next body

        self.parameter_layout.addLayout(grid_layout)

    @Slot()
    def create_new_mcmc(self) -> None:
        """Creates a new MCMC object with the specified parameters."""
        if self.raw_data_path is None:
            QMessageBox.warning(self, "Error", "Please select a raw data file.")
            return

        try:
            raw_data = np.load(self.raw_data_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading raw data: {e}")
            return

        try:
            # Collect initial parameters, param bounds, proposal std, and param names from input fields
            num_bodies = self.bodies_spinbox.value()
            initial_parameters = []
            param_bounds = []
            proposal_std = []
            param_names = self.parameter_names  # As per your request

            for body_widget in self.body_widgets:
                body_layout = body_widget.layout()
                body_initial_params = []
                body_param_bounds = []
                body_proposal_std = []

                for col in range(len(self.parameter_names)):
                    # Initial Values
                    initial_value_field = body_layout.itemAtPosition(1, col + 1).widget()
                    initial_value = float(initial_value_field.text())
                    body_initial_params.append(initial_value)

                    # Proposal Std
                    proposal_std_field = body_layout.itemAtPosition(2, col + 1).widget()
                    proposal_std_value = float(proposal_std_field.text())
                    body_proposal_std.append(proposal_std_value)

                    # Param Bounds
                    bounds_layout = body_layout.itemAtPosition(3, col + 1)
                    lower_field = bounds_layout.itemAt(0).widget()
                    upper_field = bounds_layout.itemAt(1).widget()
                    lower_bound = float(lower_field.text())
                    upper_bound = float(upper_field.text())
                    body_param_bounds.append((lower_bound, upper_bound))

                initial_parameters.append(body_initial_params)
                proposal_std.append(body_proposal_std)
                param_bounds.append(body_param_bounds)

            initial_parameters = np.array(initial_parameters)
            proposal_std = np.array(proposal_std)
            # Flatten param_bounds to a list of tuples
            param_bounds_flat = [tuple(bounds) for body_bounds in param_bounds for bounds in body_bounds]

            max_cpu_nodes = 16  # As per your previous default

            self.mcmc = MCMC(
                raw_data=raw_data,
                initial_parameters=initial_parameters,
                param_bounds=param_bounds_flat,
                proposal_std=proposal_std,
                likelihood_func=lambda x: 0.0,  # Placeholder
                param_names=param_names,
                max_cpu_nodes=max_cpu_nodes
            )
            self.console.append("Created new MCMC object.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating MCMC object: {e}")

    @Slot()
    def run_metropolis_hastings(self) -> None:
        """Runs the Metropolis-Hastings algorithm with the specified number of iterations."""
        if self.mcmc is None:
            QMessageBox.warning(self, "Error", "MCMC object is not initialized.")
            return

        num_iterations = self.iterations_spinbox.value()
        if num_iterations <= 0:
            QMessageBox.warning(self, "Error", "Number of iterations must be positive.")
            return

        self.console.append(f"Running Metropolis-Hastings for {num_iterations} iterations...")
        # Run in a separate thread to prevent GUI freezing
        threading.Thread(target=self._run_mh, args=(num_iterations,)).start()

    def _run_mh(self, num_iterations: int) -> None:
        """Internal method to run Metropolis-Hastings algorithm."""
        try:
            self.mcmc.metropolis_hastings(num_iterations)
            self.console.append(f"Completed Metropolis-Hastings for {num_iterations} iterations.")
        except Exception as e:
            self.console.append(f"Error during Metropolis-Hastings: {e}")
