#include "CommandLineParser.h"

#include <CLI/CLI.hpp>

#include <Base/Resources.h>

std::optional<int> CommandLineParser::parse(CommandLineArguments& arguments, int argc, char** argv)
{
    CLI::App app{"Command-line interface for ALIEN v" + Const::ProgramVersion};

    app.add_option("-i", arguments.inputFilename, "Specifies the name of the input file for the simulation to run.");
    app.add_option("-o", arguments.outputFilename, "Specifies the name of the output file for the simulation.");
    app.add_option(
        "-t",
        arguments.timesteps,
        "The number of time steps to be calculated. If it is not specified, the simulation runs until it is stopped with Q or Ctrl+C, which writes the "
        "output file as well.");
    app.add_option("-u,--user", arguments.userName, "The name of the user to log in to the alien server with. Requires a password to be given via -p.");
    app.add_option("-p,--password", arguments.password, "The password of the user given via -u.");
    app.add_option(
        "--upload-name",
        arguments.uploadName,
        "Enables the periodic upload of the running simulation to the alien server and specifies the base name under which it is uploaded. The final name "
        "is the base name followed by a consecutive number. The uploads are stored in the private workspace of the user given via -u and require "
        "--upload-interval to be given as well.");
    app.add_option(
        "--upload-interval", arguments.uploadInterval, "The number of minutes between two periodic uploads. Requires --upload-name to be given as well.");
    app.add_flag(
        "-d,--debug",
        arguments.debugMode,
        "Enables debug mode: this bypasses CUDA graphs and synchronizes after every kernel, so the simulation runs slower than normal but each kernel can "
        "be measured and traced individually. Two files are written: '"
            + Const::ProfileFilename.string() + "' holds the accumulated wall-clock time per kernel and '" + Const::TraceFilename.string()
            + "' holds the last kernel calls, which locates a kernel that hangs or triggers a driver timeout as the entry that is still marked as "
              "running.");
    app.add_flag(
        "--plain",
        arguments.plainOutput,
        "Disables colors, the banner and the live status panel and prints plain lines instead. This is switched on automatically when the output is "
        "redirected. Together with -t the time steps are then calculated in a single run, which makes this the most accurate mode for measuring the TPS.");

    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError const& e) {
        return app.exit(e);
    }
    return std::nullopt;
}
