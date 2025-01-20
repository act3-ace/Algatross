/*******************************************************************************
 * Set a custom icon for ruff as it's not available in the fa built-in brands
 */
var faRuff = {
    prefix: "fa-custom",
    iconName: "ruff",
    icon: [
        24, // viewBox width
        24, // viewBox height
        [], // ligature
        "e001", // unicode codepoint - private use area
        "m21.683 11.593-8.51-2.14 8.34-9.066a.23.23 0 0 0-.29-.352L2.252 11.62a.227.227 0 0 0-.108.226.23.23 0 0 0 .164.19l8.497 2.497-8.35 9.08a.228.228 0 0 0-.007.303.227.227 0 0 0 .3.047l19-11.953a.228.228 0 0 0 .105-.23.225.225 0 0 0-.172-.187z", // svg path (https://simpleicons.org/icons/ruff.svg)
    ],
};

var faUv = {
    prefix: "fa-custom",
    iconName: "uv",
    icon: [
        24, // viewBox width
        24, // viewBox height
        [], // ligature
        "e002", // unicode codepoint - private use area
        "m0 .1058.0504 11.9496.0403 9.5597c.0055 1.3199 1.08 2.3854 2.4 2.3798l9.5596-.0403 5.9749-.0252.6075-.0026c1.316-.0056 2.3799-1.0963 2.3799-2.4123h1.0946v2.3894L24 23.9042 23.8992.005 12.9056.0513l.0463 9.5245v5.9637h-1.9583L11.04 9.584 10.9936.0594Z", // svg path (https://simpleicons.org/icons/uv.svg)
    ],
};

FontAwesome.library.add(faRuff, faUv);
