<h1> Disaster Response Pipeline Project </h1>
<h3> Installation </h3>
The project requires <b> Python 3.x </b> and the following Python libraries installed:
<ul>

  <li> <a href="http://www.numpy.org/" rel="nofollow">NumPy</a> </li>
  <li> <a href="http://pandas.pydata.org" rel="nofollow">Pandas</a> </li>
  <li> <a href="https://www.nltk.org/" rel="nofollow">Natural Language Toolkit </a> </li>
  <li> <a href="https://www.nltk.org/" rel="nofollow">Natural Language Toolkit </a> </li>
  <li> <a href="https://scikit-learn.org/stable/" rel="nofollow">scikit-learn </a> </li>
  <li> <a href="https://www.sqlalchemy.org/" rel="nofollow">SQLalchemy </a> </li>
  <li> <a href="http://flask.pocoo.org/" rel="nofollow">Flask </a> </li>
  <li> <a href="https://plot.ly/python/" rel="nofollow">Plotly </a> </li>

</ul>

<h3> Project Motivation </h3>
Create a machine learning pipeline in order to categorize disaster events so messages can be
sent to the appropriate disaster relief agency.

<h3> File Descriptions </h3>
The files required to run the app are organized as followed:
<ul>
  <li> app folder </li>
    <ul>
      <li> template folder - contains master.html file which is main page of app webpage
      and go.html which is the classification result page of app </li>
      <li> run.py - Flask file which loads pickle file and runs app </li>
    </ul>
</ul>

<ul>
  <li> data folder </li>
    <ul>
      <li> disaster_categories.csv - categories data </li>
      <li> disaster_messages.csv - messages data </li>
      <li> process_data.py - process and save data to SQL Database </li>
      <li> InsertDatabaseName.db - database to save data too </li>
    </ul>
</ul>

<ul>
  <li> models folder </li>
    <ul>
      <li> train_classifier.py - create prediction pipleline with processed data created
      in SQL Database and save to pickle file </li>
      <li> classifier.pkl - saved data </li>
      </ul>

</ul>

<h3> Instructions to Run the App </h3>

<ol>
  <li> Run the following commands in the project's root directory to set up your database and model. </li>

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  <li> Run the following command in the app's directory to run your web app. </li>
    `python run.py`

  <li> Go to http://0.0.0.0:3001/ <li>
</ol>

<h3> Licensing, Authors, Acknowledgments </h3>
Credit should be given to <a target="_blank" href="https://www.figure-eight.com/">Figure Eight </a> for the data. You can find the licensing for the data and other descriptive information at the Figure Eight link provided above. The template code provided for the app (run.py), data processing (process_data.py), and training (train_classifier.py) were obtained from Udacity Data Science Nano Degree Project: Disaster Response Pipeline Project.
