import sys
import openai
import json
import time
import tempfile
from openai import OpenAI
import os

MAX_REQUESTS = 50000
MAX_FILE_SIZE_MB = 100

class BatchJobException(Exception):
    def __init__(self, error_code, message):
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.error_code}] {self.message}"

class OpenAIBatchManager:
    def __init__(self,client=None, api_key=None,name=None):
        if client is None:
            self.client =OpenAI(api_key=api_key)
        elif api_key is None:
            self.client = client
        else:
            raise Exception("Provide Open AI client or API key")
        self.name = name


    def build_batch_input_file(self, histories, model, max_tokens, file_path,json_mode=True):
        """
        Creates a .jsonl file for batch processing based on a list of histories, model name, and max tokens.

        :param histories: List of dictionaries, each containing a history of messages for a prompt.
        :param model: Name of the model to be used for the batch processing.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :param file_path: Path to the output .jsonl file.
        """

        with open(file_path, 'w') as f:
            for i, history in enumerate(histories):
                if json_mode:
                    request_data = {
                        "custom_id": f"request-{i + 1}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": history,
                            "max_tokens": max_tokens,
                            "response_format": {"type": "json_object"}
                        }
                    }
                else:
                    request_data = {
                        "custom_id": f"request-{i+1}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": history,
                            "max_tokens": max_tokens,
                        }
                    }
                f.write(json.dumps(request_data) + '\n')

        # Get file size in MB
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Check file size
        if file_size_mb > MAX_FILE_SIZE_MB:
          raise Exception(f"Batch file size is greater than {MAX_FILE_SIZE_MB} MB.")

        # Check number of lines
        with open(file_path, 'r') as file:
            lines = file.readlines()
            line_count = len(lines)

        if line_count > MAX_REQUESTS:
              raise Exception(f"Batch File has more than {MAX_REQUESTS} requests")

        with open(file_path, 'rb') as file:
            batch_input_file = self.client.files.create(
              file=file,
              purpose="batch"
            )

        batch_input_file_id = batch_input_file.id

        return batch_input_file_id



    def create_batch(self,num_request,batch_input_file_id=None,time_out=60):
        """
        Creates a batch request and manages the entire process.

        :param input_file_id: id of the uploaded input file containing batch input data.
        :param completion_window: Time window for the batch to complete.
        :return: List of results from the batch processing.
        """

        batch_id=None
        output_file_id=None
        error_file_id=None
        results = ['']*num_request
        errors = [None]*num_request
        try:

            # Create the batch using the uploaded file ID
            batch_response = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "batch processing job"
                }
            )
            batch_id = batch_response.id
            start_time = time.time()
            # Check the status of the batch periodically
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > time_out:
                    raise BatchJobException(message=f'fThe job has timed out after {int(time_out/60.0)} min. Restart the job.' , error_code="BATCH_TIME_OUT",)

                batch_job = self.client.batches.retrieve(batch_id)
                status = batch_job.status

                status_message=f'\rJob id : {batch_id};\t'
                if self.name:
                    status_message+=f'name : {self.name};\t'
                status_message+=f'status: {status}\t\t'
                sys.stdout.write(status_message)
                sys.stdout.flush()

                if status in ['completed', 'failed', 'expired', 'cancelled']:
                    print()
                    break
                time.sleep(0.5)

            # Retrieve the results if the batch is completed
            if batch_job.status == 'completed':

                output_file_id = batch_job.output_file_id
                if output_file_id:
                    results = self.retrieve_batch_results(output_file_id,num_request,True)
                else:
                    raise BatchJobException(
                    error_code="BATCH_JOB_FAILED",
                    message=f"Batch job failed. Reason: {batch_job.error_message}"
                    )
                error_file_id = batch_job.error_file_id
                if error_file_id:
                     errors = self.retrieve_batch_results(error_file_id,num_request,False)


                return {'results': results, 'errors': errors}

            elif batch_job.status == 'failed':
                raise BatchJobException(
                    error_code="BATCH_JOB_FAILED",
                    message=f"Batch job failed. Reason: {batch_job.error_message}"
                )

            elif batch_job.status == 'expired':
                raise BatchJobException(
                    error_code="BATCH_JOB_EXPIRED",
                    message="Batch job expired before completion."
                )

            elif batch_job.status == 'cancelled':
                raise BatchJobException(
                    error_code="BATCH_JOB_CANCELLED",
                    message="Batch job was cancelled by the user."
                )
        except openai.RateLimitError:
            print(f"\n Rate limit exceeded in job execution, waiting for 30 sec before trying again")
            time.sleep(30)
        except Exception as e:
            if type(e)==BatchJobException:
                raise e
            else:
                raise BatchJobException(
                    error_code="BATCH_JOB_ERROR",
                    message=f"An error occurred while processing the batch job: {str(e)}"
                )
        finally:
            if batch_id:
                try:
                    self.client.batches.cancel(batch_id)
                except Exception as e:
                    pass
            if output_file_id:
                try:
                    self.client.files.delete(output_file_id)
                except Exception as e:
                    pass
            if error_file_id:
                try:
                    self.client.files.delete(error_file_id)
                except Exception as e:
                    pass

        return {'results': results, 'errors': errors}

    def retrieve_batch_results(self, output_file_id,num_request,is_result):
        """
        Retrieves the results of a completed batch and returns them as a list.

        :param output_file_id: The ID of the output file.
        :return: List of results from the batch processing.
        """
        file_response = self.client.files.content(output_file_id).content
        if not is_result:
            results=[None]*num_request
        else:
            results = [''] * num_request

        for line in file_response.decode('utf-8').splitlines():
            result = json.loads(line)
            id = int(result['custom_id'].split('-')[1]) - 1
            try:
                #message = result.response.body.choices[0].message.content
                message = result['response']['body']['choices'][0]['message']['content']
                if message is not None:
                    results[id] = message
            except (IndexError, AttributeError, KeyError) as e:
                pass

        return results

    def batch_comprension(self, histories, model, max_tokens=4096,json_mode=True):
        """
        Main method to process a list of histories and return the results.

        :param histories: List of dictionaries, each containing a history of messages for a prompt.
        :param model: Name of the model to be used for the batch processing.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :return: List of results from the batch processing.
        """
        num_request=len(histories)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
            temp_file_path = temp_file.name


        # Build the batch input file
        batch_input_file_id=self.build_batch_input_file(histories, model, max_tokens, temp_file_path,json_mode=json_mode)

        # Create the batch and retrieve the results
        done=False
        time_out=3600
        while not done:
            try:
                results = self.create_batch(num_request,batch_input_file_id,time_out)
                done = True
            except BatchJobException as e:
                if e.error_code == 'BATCH_TIME_OUT':
                   print(f'\nBath Error:{e.message}')
                   time_out+=60
                else:
                   raise e

        if batch_input_file_id:
            try:
                self.client.files.delete(batch_input_file_id)
            except Exception as e:
                pass
        os.remove(temp_file_path)

        return results
