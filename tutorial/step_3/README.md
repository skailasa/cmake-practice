# Usage Requirements

- these are settings that propagate to consumers, which link to the target via target_link_libraries

- target_compile_definitions()
    - Adds preprocessor macros (with -D flags)

- target_compile_options()
    - Adds compiler flags, not macros, like warnings or language switches

- target_include_directories()
    - adds header search paths `-I`

- target_link_directories()
    - Adds library search paths for the linker `-L`

- target_link_options()
    - Adds linker flags


- target_precompile_headers()


target_sources()