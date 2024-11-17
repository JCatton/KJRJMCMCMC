# plots.py

"""
This file defines the PlotsPage class, which allows users to load MCMC chains and visualize the plots from the MCMC methods.

It provides an interface to select and load MCMC chains and placeholders for displaying plots.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit
)
from PySide6.QtCore import Slot
from MCMC.mcmc import MCMC
from pathlib import Path
import threading


class PlotsPage(QWidget):
    """
    The Plots page, providing controls to load MCMC chains and display plots.
    """

    def __init__(self, runtime_page):
        super().__init__()

        self.mcmc = runtime_page.mcmc  # Share the MCMC object with RuntimePage

        main_layout = QVBoxLayout(self)

        # Load MCMC chain
        load_layout = QHBoxLayout()
        self.load_label = QLabel("Load MCMC Directory:")
        self.load_button = QPushButton("Browse")
        self.load_button.clicked.connect(self.load_mcmc)
        load_layout.addWidget(self.load_label)
        load_layout.addWidget(self.load_button)
        main_layout.addLayout(load_layout)

        # Placeholder for plots
        self.plots_area = QTextEdit()
        self.plots_area.setReadOnly(True)
        self.plots_area.setText("Plots will be displayed here.")
        main_layout.addWidget(self.plots_area)

    @Slot()
    def load_mcmc(self) -> None:
        """Opens a dialog to select an existing MCMC directory and loads the MCMC object."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select MCMC Directory")
        if dir_path:
            try:
                self.mcmc = MCMC.load(Path(dir_path))
                self.plots_area.append(f"Loaded MCMC from {dir_path}")
                # Here we would update the plots accordingly
            except Exception as e:
                self.plots_area.append(f"Error loading MCMC: {e}")
