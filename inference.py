import sys
import csv
import pandas as pd
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox, QTextEdit, 
                           QProgressBar, QMessageBox, QGroupBox, QStyleFactory)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from transformers import BertForSequenceClassification, BertTokenizer

class PredictionWorker(QThread):
    progress_updated = pyqtSignal(int)
    prediction_finished = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, tokenizer, df, text_column):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.df = df.copy()
        self.text_column = text_column
        
    def run(self):
        try:
            # Create a new column for predictions
            self.df['prediction'] = None
            
            total_rows = len(self.df)
            
            for i, row in self.df.iterrows():
                # Get text from the selected column
                text = str(row[self.text_column])
                
                # Predict
                try:
                    prediction = self.predict(text)
                    self.df.at[i, 'prediction'] = prediction
                except Exception as e:
                    self.df.at[i, 'prediction'] = "Error"
                    
                # Update progress
                progress = int((i + 1) / total_rows * 100)
                self.progress_updated.emit(progress)
            
            # Return the dataframe with predictions
            self.prediction_finished.emit(self.df)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def predict(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get predicted class
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        return predicted_class

class BertPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.df = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('BERT Text Classifier')
        self.setGeometry(100, 100, 800, 600)
        
        # Set application style
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Model loading section
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        
        load_model_btn = QPushButton("Load BERT Model")
        load_model_btn.setMinimumHeight(40)
        load_model_btn.clicked.connect(self.load_model)
        
        self.model_status = QLabel("Status: Model not loaded")
        
        model_layout.addWidget(load_model_btn)
        model_layout.addWidget(self.model_status)
        model_group.setLayout(model_layout)
        
        # CSV file section
        csv_group = QGroupBox("CSV File")
        csv_layout = QVBoxLayout()
        
        csv_btn_layout = QHBoxLayout()
        load_csv_btn = QPushButton("Load CSV File")
        load_csv_btn.setMinimumHeight(40)
        load_csv_btn.clicked.connect(self.load_csv)
        
        save_csv_btn = QPushButton("Save Results")
        save_csv_btn.setMinimumHeight(40)
        save_csv_btn.clicked.connect(self.save_results)
        
        csv_btn_layout.addWidget(load_csv_btn)
        csv_btn_layout.addWidget(save_csv_btn)
        
        self.file_label = QLabel("No file selected")
        
        column_layout = QHBoxLayout()
        column_layout.addWidget(QLabel("Text Column:"))
        self.column_combo = QComboBox()
        self.column_combo.setMinimumHeight(30)
        column_layout.addWidget(self.column_combo)
        
        run_btn = QPushButton("Run Predictions")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_predictions)
        
        self.progress_bar = QProgressBar()
        
        csv_layout.addLayout(csv_btn_layout)
        csv_layout.addWidget(self.file_label)
        csv_layout.addLayout(column_layout)
        csv_layout.addWidget(run_btn)
        csv_layout.addWidget(self.progress_bar)
        csv_group.setLayout(csv_layout)
        
        # Single text prediction section
        predict_group = QGroupBox("Single Text Prediction")
        predict_layout = QVBoxLayout()
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to classify...")
        
        predict_btn = QPushButton("Predict")
        predict_btn.setMinimumHeight(40)
        predict_btn.clicked.connect(self.predict_single)
        
        self.prediction_result = QLabel("Prediction: None")
        
        predict_layout.addWidget(self.text_input)
        predict_layout.addWidget(predict_btn)
        predict_layout.addWidget(self.prediction_result)
        predict_group.setLayout(predict_layout)
        
        # Results preview section
        results_group = QGroupBox("Results Preview")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        # Add all sections to main layout
        main_layout.addWidget(model_group)
        main_layout.addWidget(csv_group)
        main_layout.addWidget(predict_group)
        main_layout.addWidget(results_group)
        
        # Apply some styling
        self._apply_styles()
        
    def _apply_styles(self):
        # Font styles
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        
        for group in self.findChildren(QGroupBox):
            group.setFont(title_font)
        
        # Button styles
        for button in self.findChildren(QPushButton):
            button.setStyleSheet("""
                QPushButton {
                    background-color: #4a86e8;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3d73c5;
                }
                QPushButton:pressed {
                    background-color: #2c5aa0;
                }
            """)
    
    def load_model(self):
        try:
            model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
            
            if model_dir:
                self.model_status.setText("Loading model...")
                QApplication.processEvents()
                
                # Load the model and tokenizer
                self.model = BertForSequenceClassification.from_pretrained(model_dir)
                self.tokenizer = BertTokenizer.from_pretrained(model_dir)
                
                # Set model to evaluation mode
                self.model.eval()
                
                self.model_status.setText(f"Status: Model loaded from {model_dir}")
                QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            self.model_status.setText("Status: Error loading model")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def load_csv(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
            
            if file_path:
                self.df = pd.read_csv(file_path)
                self.file_label.setText(f"File: {file_path}")
                
                # Populate column dropdown
                self.column_combo.clear()
                self.column_combo.addItems(self.df.columns)
                
                # Show preview
                self.update_results_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
    
    def update_results_preview(self):
        if self.df is not None:
            preview = self.df.head(5).to_string()
            self.results_text.setText(f"Preview of first 5 rows:\n\n{preview}")
    
    def run_predictions(self):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first")
            return
            
        text_column = self.column_combo.currentText()
        if not text_column:
            QMessageBox.warning(self, "Warning", "Please select a text column")
            return
        
        # Create and start worker thread
        self.worker = PredictionWorker(self.model, self.tokenizer, self.df, text_column)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.prediction_finished.connect(self.process_results)
        self.worker.error_occurred.connect(self.handle_error)
        
        # Disable UI elements during processing
        self.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def process_results(self, result_df):
        self.df = result_df
        self.update_results_preview()
        
        # Show full results
        self.results_text.setText(f"Results (showing first 20 rows):\n\n{self.df.head(20).to_string()}")
        
        QMessageBox.information(self, "Success", "Predictions completed!")
        self.setEnabled(True)
    
    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")
        self.setEnabled(True)
    
    def predict_single(self):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        text = self.text_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text")
            return
            
        try:
            # Tokenize input text
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get predicted class
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_label = "SPAM" if predicted_class != 0 else "Not SPAM"
            
            self.prediction_result.setText(f"Prediction: {predicted_label}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
    
    def save_results(self):
        if self.df is None or 'prediction' not in self.df.columns:
            QMessageBox.warning(self, "Warning", "No prediction results to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Results saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BertPredictionApp()
    window.show()
    sys.exit(app.exec())