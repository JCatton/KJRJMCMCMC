# main.py

"""
This file sets up the main GUI application for the MCMC visualization and control tool.

It creates the main window, handles navigation between pages, and initializes the application.
"""

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget
)
from PySide6.QtCore import Slot
from runtime import RuntimePage
from plots import PlotsPage
from statistics_page import StatisticsPage
from MCMC.main import gaussian_error_ln_likelihood, flux_data_from_params
import sys


class MainWindow(QMainWindow):
    """
    The main window of the application, containing navigation buttons and a stacked widget for pages.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCMC GUI")
        self.resize(1000, 800)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Navigation bar at the top
        nav_bar = QHBoxLayout()
        self.runtime_button = QPushButton("Runtime")
        self.plots_button = QPushButton("Plots")
        self.statistics_button = QPushButton("Statistics")

        self.runtime_button.clicked.connect(self.show_runtime)
        self.plots_button.clicked.connect(self.show_plots)
        self.statistics_button.clicked.connect(self.show_statistics)

        nav_bar.addWidget(self.runtime_button)
        nav_bar.addWidget(self.plots_button)
        nav_bar.addWidget(self.statistics_button)

        main_layout.addLayout(nav_bar)

        # Stacked widget to hold pages
        self.pages = QStackedWidget()
        main_layout.addWidget(self.pages)

        # Initialize pages
        self.runtime_page = RuntimePage()
        self.plots_page = PlotsPage(self.runtime_page)  # Pass runtime_page to share MCMC object
        self.statistics_page = StatisticsPage()

        self.pages.addWidget(self.runtime_page)
        self.pages.addWidget(self.plots_page)
        self.pages.addWidget(self.statistics_page)

    @Slot()
    def show_runtime(self) -> None:
        """Switches to the Runtime page."""
        self.pages.setCurrentWidget(self.runtime_page)

    @Slot()
    def show_plots(self) -> None:
        """Switches to the Plots page."""
        self.pages.setCurrentWidget(self.plots_page)

    @Slot()
    def show_statistics(self) -> None:
        """Switches to the Statistics page."""
        self.pages.setCurrentWidget(self.statistics_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
