from loguru import logger
from prefect import flow, task
from data import MRImerger
from database import TensorDatabaseManager
from Dashboard.dash_data import MRI_Dash

'''File loads data that is passed through data_url
   Augments images and creates extra dataset
   Merges original and augmented data to create final datasets with balanced classes'''


@task
def load_data(data_url):
    logger.info("load_data called")
    data_loader = MRImerger()
    return data_loader.data_merge(data_url)


@task
def dash_statistics(data):
    logger.info("dash_statistics called")
    st = MRI_Dash()
    st.dash_stats(data)
    
    
@task
def save_to_database(data):
    logger.info("save_to_database called")
    db = TensorDatabaseManager()
    db.save_dataset(data)
    

@flow
def mri_to_dataset_flow(data_url):
    logger.info("mri_to_dataset_flow called")

    # load data
    data = load_data(data_url)

    # record stats for dashboard
    dash_statistics(data)

    #save data to db
    save_to_database(data)

    return data


data_url = 'Falah/Alzheimer_MRI'

if __name__ == "__main__":
    mri_to_dataset_flow(data_url)