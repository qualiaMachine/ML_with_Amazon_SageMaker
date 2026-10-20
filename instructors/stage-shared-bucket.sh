#!/usr/bin/env bash
# Stage the shared read-only bucket for the workshop. Run once per term from a
# SageMaker notebook instance terminal (AWS CLI, Python and ~10 GB of scratch
# disk under /tmp). Idempotent: every step is a sync or a create-if-missing.
set -euo pipefail

BUCKET="ml-marathon-shared"      # the shared bucket's name
REGION="us-east-2"               # the class region
ACCOUNT_ID="123456789012"        # the 12-digit account id (top right of the console)
ROLES=("ml-sagemaker-use" "ml-sagemaker-bedrock-use")   # execution roles that may read the bucket
RELEASE_V=1                      # Badger Scribe data release version
REPO=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker
TAGS="TagSet=[{Key=Project,Value=Workshop},{Key=Name,Value=Instructor},{Key=Purpose,Value=Storage}]"

echo "== 1. bucket"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "bucket exists"
else
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
fi
aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "$TAGS"
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

PRINCIPALS=""
for r in "${ROLES[@]}"; do PRINCIPALS="$PRINCIPALS\"arn:aws:iam::$ACCOUNT_ID:role/$r\","; done
PRINCIPALS=${PRINCIPALS%,}
cat > /tmp/shared-bucket-policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": [$PRINCIPALS]},
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::$BUCKET", "arn:aws:s3:::$BUCKET/*"]
  }]
}
JSON
aws s3api put-bucket-policy --bucket "$BUCKET" --policy file:///tmp/shared-bucket-policy.json
echo "read-only policy attached for: ${ROLES[*]}"

echo "== 2. lesson repo (for data.zip and the corpus builder)"
WORK=/tmp/stage && rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"
git clone -q --depth 1 "$REPO" lesson
pip install -q pandas requests huggingface_hub

echo "== 3. WattBot: metadata, questions, PDF corpus"
unzip -q -o lesson/data/data.zip -d data
mkdir -p wattbot
cp data/data/wattbot/metadata.csv data/data/wattbot/train_QA.csv wattbot/
python lesson/scripts/build_wattbot_corpus.py --metadata wattbot/metadata.csv --out wattbot/corpus.zip
aws s3 sync wattbot/ "s3://$BUCKET/wattbot/"

echo "== 4. Badger Scribe data release v$RELEASE_V"
R="$REPO/releases/download/badger-scribe-data-v$RELEASE_V"
for a in labels images-train images-test; do
  rm -rf bs && mkdir -p bs
  curl -sSL -o "bs/$a.zip" "$R/badger-scribe-v$RELEASE_V-$a.zip"
  unzip -q -o "bs/$a.zip" -d bs/badger-scribe && rm "bs/$a.zip"
  aws s3 sync bs/badger-scribe/ "s3://$BUCKET/badger-scribe/"
  rm -rf bs
done

echo "== 5. CHURRO-3B weights (~7.5 GB)"
huggingface-cli download stanford-oval/churro-3B --local-dir churro-3B
aws s3 sync churro-3B/ "s3://$BUCKET/badger-scribe/model/churro-3B/"
rm -rf churro-3B

echo "== 6. check"
aws s3 ls "s3://$BUCKET/" --recursive --summarize | tail -2
echo "Remaining by hand: run the fine-tuning episode's homework estimator once and copy"
echo "train_summary.json, eval_before.csv, eval_after.csv from its model.tar.gz to"
echo "s3://$BUCKET/badger-scribe/instructor/full-run/"
cd / && rm -rf "$WORK"
