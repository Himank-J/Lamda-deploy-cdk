import os
from pathlib import Path
from constructs import Construct
from aws_cdk import (
    App,
    Stack,
    Environment,
    Duration,
    CfnOutput,
)
from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    Architecture,
    FunctionUrlAuthType,
)

cdk_environment = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]
)

class DogClassifierFastAPIStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "DogClassifierFastAPI",
            code=DockerImageCode.from_image_asset(
                str(Path.cwd()),  # Uses the Dockerfile in the current directory
                file="Dockerfile",
            ),
            architecture=Architecture.X86_64,
            memory_size=1092,  # 1GB memory
            timeout=Duration.minutes(5),
        )

        # Add HTTPS URL
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        # Output the function URL
        CfnOutput(
            self,
            "FunctionUrl",
            value=fn_url.url,
            description="URL for the Lambda function",
        )

app = App()
DogClassifierFastAPIStack(app, "DogClassifierFastAPIStack", env=cdk_environment)
app.synth()