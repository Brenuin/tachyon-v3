#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

struct TestRunner {
    std::string name;
    std::string command;
};

int run_test(const TestRunner& test) {
    std::cout << "Running " << test.name << "...\n";
    int code = system(test.command.c_str());
    if (code == 0) {
        std::cout << "✅ " << test.name << " passed.\n";
    } else {
        std::cout << "❌ " << test.name << " failed (code " << code << ").\n";
    }
    return code;
}

int main() {
    std::vector<TestRunner> tests = {
        {"vectort", "vectort.exe"},
        {"particlet", "particlet.exe"}
        // Add more tests here as needed
    };

    int result = 0;
    for (const auto& test : tests) {
        result |= run_test(test);
    }

    if (result == 0) {
        std::cout << "\n🎉 All tests passed!\n";
    } else {
        std::cout << "\n💥 One or more tests failed.\n";
    }

    return result;
}
