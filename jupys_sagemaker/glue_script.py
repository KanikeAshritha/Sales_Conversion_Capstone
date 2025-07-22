import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from S3
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="sales-db", 
    table_name="lead_conversion_10r_csv", 
    transformation_ctx="datasource"
)

# Write to Redshift
glueContext.write_dynamic_frame.from_options(
    frame=datasource,
    connection_type="redshift",
    connection_options={
        "redshiftTmpDir": "s3://aws-glue-assets-250260913801-ap-south-1/temporary/",
        "useConnectionProperties": "true",
        "dbtable": "public.leads_data",
        "connectionName": "Redshift connection"
    },
    transformation_ctx="redshift_output"
)

job.commit()
