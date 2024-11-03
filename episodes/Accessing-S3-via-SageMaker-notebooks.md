---
title: "Accessing and Managing Data in S3 with SageMaker Notebooks"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I load data from S3 into a SageMaker notebook?
- How do I monitor storage usage and costs for my S3 bucket?
- What steps are involved in pushing new data back to S3 from a notebook?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Read data directly from an S3 bucket into memory in a SageMaker notebook.
- Check storage usage and estimate costs for data in an S3 bucket.
- Upload new files from the SageMaker environment back to the S3 bucket.

::::::::::::::::::::::::::::::::::::::::::::::::

#### Directory setup
Let's make sure we're starting in the root directory of this instance, so that we all have our AWS_helpers.py file located in the same path (/test_AWS/scripts/AWS_helpers.py)


```python
%cd /home/ec2-user/SageMaker/
!pwd
```

    /home/ec2-user/SageMaker
    /home/ec2-user/SageMaker


## 1A. Read data from S3 into memory
Our data is stored on an S3 bucket called 'titanic-dataset-test'. We can use the following code to read data directly from S3 into memory in the Jupyter notebook environment, without actually downloading a copy of train.csv as a local file.



```python
import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

# Initialize the SageMaker role and session
# Define the SageMaker role and session
role = sagemaker.get_execution_role()
session = sagemaker.Session()
s3 = boto3.client('s3')

# Define the S3 bucket and object key
bucket_name = 'titanic-dataset-test'  # replace with your S3 bucket name

# Read the train data from S3
key = 'data/titanic_train.csv'  # replace with your object key
response = s3.get_object(Bucket=bucket_name, Key=key)
train_data = pd.read_csv(response['Body'])

# Read the test data from S3
key = 'data/titanic_test.csv'  # replace with your object key
response = s3.get_object(Bucket=bucket_name, Key=key)
test_data = pd.read_csv(response['Body'])

# check shape
print(train_data.shape)
print(test_data.shape)

# Inspect the first few rows of the DataFrame
train_data.head()
```

    sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
    sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
    (712, 12)
    (179, 12)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>693</td>
      <td>1</td>
      <td>3</td>
      <td>Lam, Mr. Ali</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1601</td>
      <td>56.4958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>528</td>
      <td>0</td>
      <td>1</td>
      <td>Farthing, Mr. John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17483</td>
      <td>221.7792</td>
      <td>C95</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>856</td>
      <td>1</td>
      <td>3</td>
      <td>Aks, Mrs. Sam (Leah Rosen)</td>
      <td>female</td>
      <td>18.0</td>
      <td>0</td>
      <td>1</td>
      <td>392091</td>
      <td>9.3500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>802</td>
      <td>1</td>
      <td>2</td>
      <td>Collyer, Mrs. Harvey (Charlotte Annie Tate)</td>
      <td>female</td>
      <td>31.0</td>
      <td>1</td>
      <td>1</td>
      <td>C.A. 31921</td>
      <td>26.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## 1B. Download copy into notebook environment
If you have larger dataset (> 1GB), you may want to skip this step and always read directly into memory. However, for smaller datasets, it can be convenient to have a "local" copy (i.e., one stored in your notebook's instance).

Download data from S3 to notebook environment. You may need to hit refresh on the file explorer panel to the left to see this file. If you get any permission issues...

* check that you have selected the appropriate policy for this notebook
* check that your bucket has the appropriate policy permissions


```python
# Define the S3 bucket and file location
file_key = "data/titanic_train.csv"  # Path to your file in the S3 bucket
local_file_path = "./titanic_train.csv"  # Local path to save the file

# Initialize the S3 client and download the file
s3 = boto3.client("s3")
s3.download_file(bucket_name, file_key, local_file_path)
print("File downloaded:", local_file_path)
```

    File downloaded: ./titanic_train.csv


## 2. Check current size and storage costs of bucket

It's a good idea to periodically check how much storage you have used in your bucket. You can do this from a Jupyter notebook in SageMaker by using the **Boto3** library, which is the AWS SDK for Python. This will allow you to calculate the total size of objects within a specified bucket. Here’s how you can do it...

### Step 1: Set Up the S3 Client and Calculate Bucket Size


```python
# Initialize the total size counter
total_size_bytes = 0

# List and sum the size of all objects in the bucket
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket_name):
    for obj in page.get('Contents', []):
        total_size_bytes += obj['Size']

# Convert the total size to gigabytes for cost estimation
total_size_gb = total_size_bytes / (1024 ** 3)
# print(f"Total size of bucket '{bucket_name}': {total_size_gb:.2f} GB") # can uncomment this if you want GB reported

# Convert the total size to megabytes for readability
total_size_mb = total_size_bytes / (1024 ** 2)
print(f"Total size of bucket '{bucket_name}': {total_size_mb:.2f} MB")
```

    Total size of bucket 'titanic-dataset-test': 41.04 MB


We have added this code to a helper called `get_s3_bucket_size(bucket_name)` for your convenience. You can call this function via the below code.


```python
import test_AWS.scripts.AWS_helpers as helpers # test_AWS.scripts.AWS_helpers reflects path leading up to AWS_helpers.py

helpers.get_s3_bucket_size(bucket_name)
```




    {'size_mb': 41.043779373168945, 'size_gb': 0.0400818157941103}



### Explanation

1. **Paginator**: Since S3 buckets can contain many objects, we use a paginator to handle large listings.
2. **Size Calculation**: We sum the `Size` attribute of each object in the bucket.
3. **Unit Conversion**: The size is given in bytes, so dividing by `1024 ** 2` converts it to megabytes (MB).

> **Note**: If your bucket has very large objects or you want to check specific folders within a bucket, you may want to refine this code to only fetch certain objects or folders.

## 3: Check storage costs of bucket
To estimate the storage cost of your Amazon S3 bucket directly from a Jupyter notebook in SageMaker, you can use the following approach. This method calculates the total size of the bucket and estimates the monthly storage cost based on AWS S3 pricing.

**Note**: AWS S3 pricing varies by region and storage class. The example below uses the S3 Standard storage class pricing for the US East (N. Virginia) region as of November 1, 2024. Please verify the current pricing for your specific region and storage class on the [AWS S3 Pricing page](https://aws.amazon.com/s3/pricing/).




```python
# AWS S3 Standard Storage pricing for US East (N. Virginia) region
# Pricing tiers as of November 1, 2024
first_50_tb_price_per_gb = 0.023  # per GB for the first 50 TB
next_450_tb_price_per_gb = 0.022  # per GB for the next 450 TB
over_500_tb_price_per_gb = 0.021  # per GB for storage over 500 TB

# Calculate the cost based on the size
if total_size_gb <= 50 * 1024:
    # Total size is within the first 50 TB
    cost = total_size_gb * first_50_tb_price_per_gb
elif total_size_gb <= 500 * 1024:
    # Total size is within the next 450 TB
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           ((total_size_gb - 50 * 1024) * next_450_tb_price_per_gb)
else:
    # Total size is over 500 TB
    cost = (50 * 1024 * first_50_tb_price_per_gb) + \
           (450 * 1024 * next_450_tb_price_per_gb) + \
           ((total_size_gb - 500 * 1024) * over_500_tb_price_per_gb)

print(f"Estimated monthly storage cost: ${cost:.4f}")


```

    Estimated monthly storage cost: $0.0009


For your convenience, we have also added this code to a helper function.


```python
import test_AWS.scripts.AWS_helpers as helpers # test_AWS.scripts.AWS_helpers reflects path leading up to AWS_helpers.py

cost = helpers.calculate_s3_storage_cost(bucket_name)

print(f"Estimated monthly storage cost for bucket '{bucket_name}': ${cost:.4f}")

```

    Estimated monthly storage cost for bucket 'titanic-dataset-test': $0.0009


**Important Considerations**:

- **Pricing Tiers**: AWS S3 pricing is tiered. The first 50 TB per month is priced at `$0.023 per GB`, the next 450 TB at `$0.022 per GB`, and storage over 500 TB at `$0.021 per GB`. Ensure you apply the correct pricing tier based on your total storage size.
- **Region and Storage Class**: Pricing varies by AWS region and storage class. The example above uses the S3 Standard storage class pricing for the US East (N. Virginia) region. Adjust the pricing variables if your bucket is in a different region or uses a different storage class.
- **Additional Costs**: This estimation covers storage costs only. AWS S3 may have additional charges for requests, data retrievals, and data transfers. For a comprehensive cost analysis, consider these factors as well.

For detailed and up-to-date information on AWS S3 pricing, please refer to the [AWS S3 Pricing page](https://aws.amazon.com/s3/pricing/).



## 4. Pushing new files from notebook environment to bucket
As your analysis generates new files, you can upload to your bucket as demonstrated below. For this demo, you can create a blank `results.txt` file to upload to your bucket.


```python
# Define the S3 bucket name and the file paths
train_file_path = "results.txt"

# Upload the training file
s3.upload_file(train_file_path, bucket_name, "results/results.txt")

print("Files uploaded successfully.")

```

    Files uploaded successfully.


# Developer use - produce md version of this notebook


```python
!pwd
!jupyter nbconvert --to markdown test_AWS/Accessing-S3-via-SageMaker-notebooks.ipynb

```

    /home/ec2-user/SageMaker
    [NbConvertApp] WARNING | pattern 'test_AWS/Data-storage-and-access-via-buckets.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    


:::::::::::::::::::::::::::::::::::::: keypoints 

- Load data from S3 into memory for efficient storage and processing.
- Periodically check storage usage and costs to manage S3 budgets.
- Use SageMaker to upload analysis results and maintain an organized workflow.

::::::::::::::::::::::::::::::::::::::::::::::::
