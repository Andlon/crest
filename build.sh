SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p $SCRIPTDIR/target
mkdir -p $SCRIPTDIR/target/debug
mkdir -p $SCRIPTDIR/target/release

printf "Building debug binaries\n"
printf "***********************\n"
cd $SCRIPTDIR/target/debug
cmake -DCMAKE_BUILD_TYPE=Debug $SCRIPTDIR
make -j4
printf "Completed debug binaries.\n"

printf "\n\n\n"

printf "Building release binaries\n"
printf "*************************\n"
cd $SCRIPTDIR/target/release
cmake -DCMAKE_BUILD_TYPE=Release $SCRIPTDIR
make -j4
printf "Completed release binaries.\n"

printf "\n\n"
printf "Completed full build.\n"
