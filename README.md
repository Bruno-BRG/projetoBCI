# EEG Motor Imagery Classification GUI Application

This application converts the Jupyter notebook EEG classifier into a desktop GUI application using PyQt5. It provides an intuitive interface for loading EEG data, training deep learning models, and making real-time predictions for motor imagery classification.

## Features

- **Data Loading**: Load and preprocess EEG data from the PhysioNet dataset
- **Interactive Visualization**: Real-time plotting of EEG signals and training metrics
- **Model Training**: Train CNN+Transformer models with configurable parameters
- **Real-time Prediction**: Classify motor imagery (left/right hand movement) in real-time
- **Model Management**: Save and load trained models
- **Dark Theme UI**: Modern, user-friendly interface

## Installation

1. **Clone or download the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the EEG data:**
   - The application can automatically download PhysioNet data
   - Or you can specify a custom data directory

## Usage

### Running the Application

```bash
python eeg_classifier_app.py
```

### Application Tabs

#### 1. Data Tab
- **Load Data**: Browse and select your EEG data directory
- **Configure Subjects**: Set the number of subjects to load (1-109)
- **Visualize**: Interactive slider to browse through EEG samples
- **Monitor**: View data loading progress and statistics

#### 2. Training Tab
- **Configure Training**:
  - Max Epochs: Number of training epochs (default: 100)
  - Batch Size: Training batch size (default: 10)
  - Learning Rate: Optimizer learning rate (default: 0.0005)
  - Dropout: Regularization dropout rate (default: 0.125)
- **Start Training**: Begin model training with real-time progress monitoring
- **View Progress**: Live plots of training/validation loss and accuracy curves

#### 3. Prediction Tab
- **Load Models**: Load previously saved model checkpoints
- **Real-time Prediction**: Classify random EEG samples
- **Results Visualization**: View prediction results with confidence scores
- **Model Management**: Save trained models for future use

## Model Architecture

The application uses a hybrid CNN+Transformer architecture:

- **CNN Layers**: Extract local temporal features from EEG signals
- **Transformer Blocks**: Capture long-range dependencies with self-attention
- **Classification Head**: Binary classification for left/right motor imagery

## Data Format

The application expects EEG data in the PhysioNet EEG Motor Movement/Imagery Dataset format:
- **Channels**: 64 EEG electrodes
- **Sampling Rate**: 160 Hz
- **Tasks**: Motor imagery (left hand vs. right hand)
- **Format**: EDF files from PhysioNet

## File Structure

```
eeg_classifier_app.py    # Main application GUI
eeg_model.py            # Neural network models and training logic
data_processor.py       # EEG data loading and preprocessing
requirements.txt        # Python dependencies
README.md              # This file
```

## Customization

### Adding New Models
1. Implement your model in `eeg_model.py`
2. Update the model selection in the GUI
3. Modify training parameters as needed

### Custom Data Sources
1. Modify `data_processor.py` to handle your data format
2. Update the data loading interface in the GUI
3. Ensure compatibility with the model input requirements

### UI Modifications
1. Edit `eeg_classifier_app.py` to modify the interface
2. Add new tabs, controls, or visualizations
3. Customize the dark theme styling

## Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Ensure MNE can access the PhysioNet dataset
   - Check file permissions and network connectivity
   - Verify data path is correct

2. **Training Failures**:
   - Check GPU/CPU availability
   - Ensure sufficient memory for batch size
   - Verify data preprocessing completed successfully

3. **UI Issues**:
   - Update PyQt5 to latest version
   - Check matplotlib backend compatibility
   - Ensure all dependencies are installed

### Performance Tips

1. **Faster Training**:
   - Use GPU acceleration if available
   - Reduce batch size if memory is limited
   - Start with fewer subjects for testing

2. **Better Accuracy**:
   - Increase number of training epochs
   - Experiment with different learning rates
   - Try data augmentation techniques

## Dependencies

- **PyQt5**: GUI framework
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training framework
- **MNE**: EEG data processing
- **Matplotlib**: Plotting and visualization
- **NumPy**: Numerical computing

## License

This project is based on the EEG classification notebook and is intended for educational and research purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Citation

If you use this application in your research, please cite the original PhysioNet EEG dataset and any relevant papers.
