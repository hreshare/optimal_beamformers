#export KALDI_ROOT=`pwd`/../../..
export KALDI_ROOT=/misc/export3/hiroe/work2022/kaldi
[ -f $KALDI_ROOT/tools/extras/env.sh ] && . $KALDI_ROOT/tools/extras/env.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
