module.exports = {
    pathPrefix: `/eyeofgollum`,
    plugins: [
        {
            resolve: `gatsby-plugin-typography`,
            options: {
                pathToConfigModule: `src/utils/typography.js`,
            },
        },
    ],
};