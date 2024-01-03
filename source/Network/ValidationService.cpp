#include "ValidationService.h"

bool ValidationService::isStringValidForDatabase(std::string const& s)
{
    for (char const& ch : s) {
        if (!std::isalnum(ch) && ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t' && ch != '-' && ch != '+' && ch != '_' && ch != '/' && ch != '*'
            && ch != '~' && ch != '#' && ch != '.' && ch != ':' && ch != ',' && ch != '<' && ch != '>' && ch != '|' && ch != '!' && ch != '&' && ch != '('
            && ch != ')' && ch != '[' && ch != ']' && ch != '{' && ch != '}' && ch != '?') {
            return false;
        }
    }
    return true;
}
