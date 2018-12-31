<h1> Disaster Response Pipeline Project </h1>
<h3> Installation </h3>
The project requires <b> Python 3.x </b> and the following Python libraries installed:
<ul>
  <li> <a href="http://www.numpy.org/" rel="nofollow">NumPy</a> </li>
  <li> <a href="http://pandas.pydata.org" rel="nofollow">Pandas</a> </li>
  <li> <a href="https://www.nltk.org/" rel="nofollow">Natural Language Toolkit </a> </li>
  <li> <a href="https://www.nltk.org/" rel="nofollow">Natural Language Toolkit </a> </li>
  <li> <a href="https://docs.python.org/2/library/pickle.html" rel="nofollow">pickle </a> </li>

  
</ul>

<h3> Project Description </h3>


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
