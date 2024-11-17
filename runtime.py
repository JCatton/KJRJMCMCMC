# runtime.py

"""
This file defines the RuntimePage class, which provides the interface for loading and running MCMC simulations.

It allows the user to load an existing MCMC object, create a new one with specified parameters, and run the Metropolis-Hastings algorithm.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QSpinBox, QFormLayout, QTextEdit, QMessageBox,
)
from PySide6.QtCore import Slot
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

        main_layout = QHBoxLayout(self)

        # Left side: Load existing MCMC or create new one
        left_layout = QVBoxLayout()

        # Load existing MCMC
        load_layout = QHBoxLayout()
        self.load_label = QLabel("Load MCMC Directory:")
        self.load_button = QPushButton("Browse")
        self.load_button.clicked.connect(self.load_mcmc)
        load_layout.addWidget(self.load_label)
        load_layout.addWidget(self.load_button)
        left_layout.addLayout(load_layout)

        # Create new MCMC form
        self.form_layout = QFormLayout()
        self.raw_data_button = QPushButton("Select Raw Data File")
        self.raw_data_button.clicked.connect(self.select_raw_data)
        self.raw_data_path = None
        self.form_layout.addRow("Raw Data:", self.raw_data_button)

        self.initial_parameters_input = QLineEdit()
        self.form_layout.addRow("Initial Parameters:", self.initial_parameters_input)

        self.param_bounds_input = QLineEdit()
        self.form_layout.addRow("Parameter Bounds:", self.param_bounds_input)

        self.proposal_std_input = QLineEdit()
        self.form_layout.addRow("Proposal Std Dev:", self.proposal_std_input)

        self.param_names_input = QLineEdit()
        self.form_layout.addRow("Parameter Names:", self.param_names_input)

        self.max_cpu_nodes_input = QSpinBox()
        self.max_cpu_nodes_input.setMinimum(1)
        self.max_cpu_nodes_input.setMaximum(64)
        self.max_cpu_nodes_input.setValue(16)
        self.form_layout.addRow("Max CPU Nodes:", self.max_cpu_nodes_input)

        self.create_mcmc_button = QPushButton("Create New MCMC")
        self.create_mcmc_button.clicked.connect(self.create_new_mcmc)
        self.form_layout.addRow(self.create_mcmc_button)

        left_layout.addLayout(self.form_layout)

        main_layout.addLayout(left_layout)

        # Right side: Run Metropolis-Hastings and IPython console
        right_layout = QVBoxLayout()

        # Run Metropolis-Hastings
        rh_layout = QHBoxLayout()
        self.iterations_label = QLabel("Number of New Iterations:")
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setMinimum(1)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_metropolis_hastings)
        rh_layout.addWidget(self.iterations_label)
        rh_layout.addWidget(self.iterations_spinbox)
        rh_layout.addWidget(self.run_button)
        right_layout.addLayout(rh_layout)

        # IPython console placeholder (actual implementation would require additional packages)
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
                self.proposal_std_input.setText(str(self.mcmc.proposal_std))
                self.initial_parameters_input.setText(str(self.mcmc.initial_parameters))
                self.param_names_input.setText(str(self.mcmc.param_names))
                self.param_bounds_input.setText(str(self.mcmc.param_bounds))
                self.max_cpu_nodes_input.setValue(self.mcmc.max_cpu_nodes)
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
            initial_parameters = np.array(eval(self.initial_parameters_input.text()))
            param_bounds = eval(self.param_bounds_input.text())
            proposal_std = np.array(eval(self.proposal_std_input.text()))
            param_names = self.param_names_input.text().split(',')
            max_cpu_nodes = self.max_cpu_nodes_input.value()

            self.mcmc = MCMC(
                raw_data=raw_data,
                initial_parameters=initial_parameters,
                param_bounds=param_bounds,
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
