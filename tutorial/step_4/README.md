# Generator Expressions

- evaluated during build, produce specific config info

- Allowed in the context of many target properties such as link libraries, include directories and compile definitions.

- - May be used to enable conditional linking, definitions, conditional directory inclusions.

- - common use is to conditionally add compiler flags, such as those for language levels or warnings
- a godo pattern is to associate this information to an interface target allowing this information to propagate.

