#!/usr/bin/env python
from __future__ import print_function, division
import sys
import re
import os.path
from textwrap import dedent

escape_re = re.compile(r'[\\"]')

def escape(s):
    return escape_re.sub(r'\\\g<0>', s)

def main(argv):
    if len(argv) < 2:
        print("Usage: {0} <input.cl>... <output.cpp>".format(sys.argv[0]), file = sys.stderr)
        return 2

    with open(sys.argv[-1], 'w') as outf:
        print(dedent('''
            #include <common/debug_unordered_map.h>
            #include <common/debug_string.h>
            #include <utility>
            #include <stdexcept>

            static const uts::unordered_map<uts::string, uts::string> g_sources{
            '''), file = outf)
        for i in sys.argv[1:-1]:
            label = os.path.basename(i)
            with open(i, 'r') as inf:
                lines = inf.readlines()
                lines = [escape(line.rstrip('\r\n')) for line in lines]
            print('    std::pair<uts::string, uts::string>("{0}",'.format(escape(label)), file = outf)
            for line in lines:
                print('        "{0}\\n"'.format(line), file = outf)
            print('    ),', file = outf)
        print(dedent('''
            };

            const uts::unordered_map<uts::string, uts::string> &getSourceMap()
            {
                return g_sources;
            }

            const uts::string &getSource(const uts::string &filename)
            {
                auto pos = g_sources.find(filename);
                if (pos == g_sources.end())
                    throw std::invalid_argument("Source " + filename + " not found");
                return pos->second;
            }
            '''), file = outf)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
