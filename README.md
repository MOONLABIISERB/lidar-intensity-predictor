# LiDAR-Intensity-based-predictor
##### Accepted for publication to Special Proceedings in Advanced Robotics at International Symposium on Experimental Robotics 2023.
##### Repository under frequent updation. For Queries contact kasiv@tamu.edu.
 ## Training data and ground truth extractor.
 <pre>
 python intensity_analyser.py
 </pre>
 Extracts the data and calibrates then for training, generating ground truth (Intensity ranges of classes), plotting etc.
 
 <pre>
 python /utils/GT_corrector.py
 </pre>
 To correct the outliers in the ground truth and data generated from <intensity_analyser.py>
 ## Angle of Incidence predictor
 
 <pre>
 python /alpha_predictor/alpha_model.py
 </pre>
 Contains the ANN architecture of the predictor.
 
 <pre>
 python /alpha_predictor/train_alpha.py
 </pre>
 Command to train the model. Please edit the paths to the root file of the dataset.
 
 ## LiDAR Semantic predictor
 ### Predictor for Ouster
 <pre>
 python intensity_predictor.py
 </pre>
 ### Predictor for Velodyne
 <pre>
  python intensity_predictor_velodyne.py
 </pre>
 ### Generate point cloud(intensity replaced by reflectivity). 
 <pre>
  python ins2ref.py
 </pre>
 
 Command to predict the classes of the LiDAR points. Reads .ply files and predicts for classes: grass, bush, trees, puddle, person.
 
 ## Utils
 <pre>
 python /utils/make_video.py
 python /utils/data_counter.py
 </pre>
 To generate movies from images.
