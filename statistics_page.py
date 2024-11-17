# statistics_page.py

"""
This file defines the StatisticsPage class, which allows users to load multiple MCMC chains and display statistical information.

It provides an interface to select and load MCMC chains, calculates the Gelman-Rubin statistic, and displays individual chain statistics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QTextEdit, QComboBox, QListWidget, QMessageBox
)
from PySide6.QtCore import Slot
from MCMC.mcmc import Statistics
from pathlib import Path


class StatisticsPage(QWidget):
    """
    The Statistics page, providing controls to load MCMC chains and display statistical information.
    """

    def __init__(self):
        super().__init__()

        self.statistics = None  # Placeholder for the Statistics object
        self.mcmc_dirs = []  # List of MCMC directories

        main_layout = QHBoxLayout(self)

        # Left side: Load MCMC chains and display Gelman-Rubin value
        left_layout = QVBoxLayout()

        # Buttons to add and remove MCMC chains
        button_layout = QHBoxLayout()
        self.add_chain_button = QPushButton("Add MCMC Chain")
        self.add_chain_button.clicked.connect(self.add_mcmc_chain)
        self.remove_chain_button = QPushButton("Remove Selected Chain")
        self.remove_chain_button.clicked.connect(self.remove_mcmc_chain)
        button_layout.addWidget(self.add_chain_button)
        button_layout.addWidget(self.remove_chain_button)
        left_layout.addLayout(button_layout)

        # List of MCMC chains
        self.chain_list = QListWidget()
        left_layout.addWidget(self.chain_list)

        # Calculate Gelman-Rubin value
        self.calc_gr_button = QPushButton("Calculate Gelman-Rubin")
        self.calc_gr_button.clicked.connect(self.calculate_gelman_rubin)
        left_layout.addWidget(self.calc_gr_button)

        self.gr_label = QLabel("Gelman-Rubin: N/A")
        left_layout.addWidget(self.gr_label)

        # Chain selector
        self.chain_selector = QComboBox()
        self.chain_selector.currentIndexChanged.connect(self.update_chain_statistics)
        left_layout.addWidget(self.chain_selector)

        # Display burn-in index and parameter statistics
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        left_layout.addWidget(self.stats_text)

        main_layout.addLayout(left_layout)

        # Right side: Two placeholder widgets
        right_layout = QVBoxLayout()
        self.placeholder1 = QLabel("Placeholder Widget 1")
        self.placeholder2 = QLabel("Placeholder Widget 2")
        right_layout.addWidget(self.placeholder1)
        right_layout.addWidget(self.placeholder2)

        main_layout.addLayout(right_layout)

    @Slot()
    def add_mcmc_chain(self) -> None:
        """Allows the user to add an MCMC chain directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select MCMC Directory")
        if dir_path:
            self.mcmc_dirs.append(dir_path)
            self.chain_list.addItem(dir_path)
            self.chain_selector.addItem(Path(dir_path).name)

    @Slot()
    def remove_mcmc_chain(self) -> None:
        """Removes the selected MCMC chain from the list."""
        selected_items = self.chain_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "No chain selected.")
            return
        for item in selected_items:
            idx = self.chain_list.row(item)
            self.chain_list.takeItem(idx)
            del self.mcmc_dirs[idx]
            self.chain_selector.removeItem(idx)

    @Slot()
    def calculate_gelman_rubin(self) -> None:
        """Calculates the Gelman-Rubin statistic for the loaded MCMC chains."""
        if not self.mcmc_dirs:
            self.gr_label.setText("Gelman-Rubin: N/A (No chains loaded)")
            return
        try:
            self.statistics = Statistics(self.mcmc_dirs)
            gr_values = self.statistics.calc_gelman_rubin()
            self.gr_label.setText(f"Gelman-Rubin: {gr_values}")
            self.update_chain_statistics()
        except Exception as e:
            self.stats_text.append(f"Error calculating Gelman-Rubin: {e}")

    @Slot()
    def update_chain_statistics(self) -> None:
        """Updates the displayed statistics for the selected chain."""
        if self.statistics is None:
            return

        chain_idx = self.chain_selector.currentIndex()
        if chain_idx < 0 or chain_idx >= len(self.statistics.loaded_mcmcs):
            return

        mcmc = self.statistics.loaded_mcmcs[chain_idx]
        burn_in_index = mcmc.burn_in_index
        param_names = mcmc.param_names
        means = mcmc.mean
        variances = mcmc.var

        stats_info = f"Burn-in Index: {burn_in_index}\n"
        stats_info += "Parameter Statistics:\n"
        for name, mean, var in zip(param_names, means, variances):
            stats_info += f"{name}: Mean = {mean}, Variance = {var}\n"

        self.stats_text.setText(stats_info)
