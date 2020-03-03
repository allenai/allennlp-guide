// This component is a generalized navigation menu used for chapter outline side-nav on desktop
// and global navigation on mobile. Based on menu design and implementation on AI2 website:
// https://github.com/allenai/ai2-web/blob/master/ui/lib/components/chrome/Header.tsx

import React from 'react';
import styled, { createGlobalStyle, css, keyframes } from 'styled-components';
import { Menu, Icon } from 'antd';
import { Link } from '../components/Link';
import { outline } from '../outline';
import { getIcon } from '../utils';

export const Navigation = ({
    className,
    isMobile = false,
    headerLinks,
    groupedChapters,
    defaultSelectedKeys = [],
    defaultOpenKeys = [],
    onTitleClick = () => {}
}) => {
    // Default Desktop components
    let MenuContainer = DesktopMenu;
    let MenuItem = Menu.Item;
    let SubMenu = Menu.SubMenu;
    let SubMenuItem = Menu.Item;
    let SubMenuTitle = DesktopSubMenuTitle;
    let LandingLink = Link;
    let InteriorLink = Link;
    // Mobile components
    if (isMobile) {
        MenuContainer = Menu;
        MenuItem = MobileMenuItem;
        SubMenu = MobileSubMenu;
        SubMenuItem = MobileSubMenuItem;
        SubMenuTitle = MobileSubMenuTitle;
        LandingLink = MobileLandingLink;
        InteriorLink = MobileInteriorLink;
    }

    const getMenuIcon = obj =>
        obj.antMenuIcon ? (
            <Icon type={obj.antMenuIcon} />
        ) : (
            obj.icon && <CustomIcon component={() => getIcon(obj.icon, 17)} />
        );

    return (
        <React.Fragment>
            <GlobalDesktopStyle />
            <MenuContainer
                className={className}
                defaultSelectedKeys={defaultSelectedKeys}
                defaultOpenKeys={defaultOpenKeys}
                mode="inline"
                inlineIndent={isMobile ? 0 : 24}>
                {isMobile && (
                    <MenuItem key="home">
                        <LandingLink href="/">Home</LandingLink>
                    </MenuItem>
                )}
                <MenuItem key={outline.overview.slug}>
                    <LandingLink to={outline.overview.slug}>
                        {!isMobile && getMenuIcon(outline.overview)}
                        {groupedChapters[outline.overview.slug].node.frontmatter.title}
                    </LandingLink>
                </MenuItem>
                {outline.parts.map(
                    part =>
                        part.chapterSlugs && (
                            <SubMenu
                                key={part.title}
                                onTitleClick={() => onTitleClick && onTitleClick(part.title)}
                                title={
                                    <SubMenuTitle>
                                        {!isMobile && getMenuIcon(part)}
                                        {part.title}
                                    </SubMenuTitle>
                                }>
                                {part.chapterSlugs.map(chapterSlug => (
                                    <SubMenuItem key={chapterSlug}>
                                        <InteriorLink to={chapterSlug}>
                                            {groupedChapters[chapterSlug].node.frontmatter.title}
                                        </InteriorLink>
                                    </SubMenuItem>
                                ))}
                            </SubMenu>
                        )
                )}
                {isMobile &&
                    headerLinks.map(headerLink => (
                        <MenuItem key={headerLink.url}>
                            <LandingLink to={headerLink.url}>{headerLink.text}</LandingLink>
                        </MenuItem>
                    ))}
            </MenuContainer>
        </React.Fragment>
    );
};

// Keyframe data for mobile nav entrance animation
export const mobileNavEntrance = yOffset => keyframes`
    0% {
        opacity: 0;
        // Passing y offset value from where this animation is called
        transform: translateY(-${yOffset});
    }
    50% {
        transform: translateY(0);
    }
    100% {
        opacity: 1;
    }
`;

const DesktopMenu = styled(Menu)``;
const DesktopSubMenuTitle = styled.span``;
const CustomIcon = styled(Icon)``;

// Resetting Ant Menu Styles
const GlobalDesktopStyle = createGlobalStyle`
    &&& {
        .ant-menu {
            border: none !important;

            svg {
                color: ${({ theme }) => theme.color.N8};
            }

            .ant-menu-submenu-title,
            .ant-menu-item {
                overflow: visible !important;
                white-space: normal !important;
                height: auto !important;
                line-height: 1.5 !important;
            }

            .ant-menu-item {
                a {
                    color: ${({ theme }) => theme.color.N10};
                }

                &.ant-menu-item-selected {
                    background: ${({ theme }) => theme.color.B1} !important;

                    &:after {
                        border-color: ${({ theme }) => theme.color.B5} !important;
                    }

                    a {
                        color: ${({ theme }) => theme.color.B5} !important;
                    }
                }
            }
        }

        ${DesktopMenu} {
            ${CustomIcon} {
                svg {
                    width: 17px;
                    height: 17px;
                    margin-right: -4px;
                    transform: translate(-2px, 1.5px);
                    stroke: ${({ theme }) => theme.color.N8};
                }
            }

            .ant-menu-submenu {
                border-top: 1px solid ${({ theme }) => theme.color.N4} !important;

                &.ant-menu-submenu-selected {
                    &,
                    span,
                    i,
                    svg {
                        color: ${({ theme }) => theme.color.B5} !important;
                        stroke: ${({ theme }) => theme.color.B5};
                    }
                }

                .ant-menu-submenu-title:hover {
                    &,
                    span,
                    i,
                    svg,
                    i:before,
                    i:after {
                        color: ${({ theme }) => theme.color.B5} !important;
                        stroke: ${({ theme }) => theme.color.B5};
                    }
                }
            }

            .ant-menu-submenu-title {
                &:hover {
                    .ant-menu-submenu-arrow {
                        &:before,
                        &:after {
                            background: linear-gradient(90deg, ${({ theme }) =>
                                `${theme.color.B5}, ${theme.color.B5}`}) !important;
                        }
                    }
                }
            }

            .ant-menu-submenu-title,
            .ant-menu-item {
                padding-top: 9px !important;
                padding-bottom 10px !important;
            }

            .ant-menu-item {
                a {
                    &:hover {
                        &,
                        svg {
                            color: ${({ theme }) => theme.color.B5};
                            stroke: ${({ theme }) => theme.color.B5};
                        }
                        text-decoration: none;
                    }
                }
            }
        }
    }
`;

const linkStyles = () => css`
    ${({ theme }) => theme.typography.body}
    display: block;
    font-weight: ${({ theme }) => theme.typography.fontWeightMedium};
    color: ${({ theme }) => theme.color.B6};
    padding: ${({ theme }) => theme.spacing.xs};
`;

const landingLinkStyles = () => css`
    ${linkStyles}
    color: ${({ theme }) => theme.color.N9};
`;

const MobileInteriorLink = styled(Link)`
    transition: none;

    &:active {
        text-decoration: none;
    }

    ${linkStyles}

    &&:hover {
        color: ${({ theme }) => theme.color.B6};
    }
`;

const MobileLandingLink = styled(MobileInteriorLink)`
    ${landingLinkStyles}
`;

const MobileSubMenuTitle = styled(DesktopSubMenuTitle)`
    ${landingLinkStyles}
    cursor: default;
`;

// These styles are applied both to MenuItem and
// SubMenu components
const mobileNavItemStyles = () => css`
    // horizontal divider line between top-level items
    border-top: 1px solid ${({ theme }) => theme.color.N4};

    // Give all top-level links, section titles and interior links
    // consistent look and feel per design
    ${MobileSubMenuTitle},
    ${MobileLandingLink},
    ${MobileInteriorLink} {
        ${({ theme }) => theme.typography.bodyBig};
        padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.lg}`};
        color: ${({ theme }) => theme.color.N9};
        text-decoration: none;
    }

    // Give mobile nav links an obvious tap state
    // (MobileSubMenuTitle doesn't apply since it's not clickable)
    ${MobileLandingLink},
    ${MobileInteriorLink} {
        -webkit-tap-highlight-color: ${({ theme }) => theme.color.B6};
    }
`;

// These are styles applied to the direct parent of MobileSubMenuTitle, MobileLandingLink,
// or MobileInteriorLink. SubMenu structures have an additional node between
// top-level item li and link which is why these can't be applied to
// `mobileNavItemStyles` above
const mobileNavLinkContainerStyles = () => css`
    // Reset hard-coded Ant spacing
    margin: 0;
    height: auto;
    line-height: inherit;
`;

// Disable hard-coded Ant menu background and pipe styling
const antSelectedItemOverrideStyles = () => css`
    &,
    &:active,
    &:hover {
        background: transparent;
    }

    // Right-aligned nav item pipe
    &:after {
        border-right-color: transparent;
    }
`;

// Mobile Nav Item that does not have a submenu
const MobileMenuItem = styled(Menu.Item)`
    ${mobileNavItemStyles}

    &&& {
        margin-bottom: 0 !important;
        ${mobileNavLinkContainerStyles}
    }
`;

// Mobile Nav Item that has a submenu
const MobileSubMenu = styled(Menu.SubMenu)`
    ${mobileNavItemStyles}

    &&& {
        // Giving expand/collapse trigger a wider hit target
        .ant-menu-submenu-title {
            ${mobileNavLinkContainerStyles}
            padding-right: ${({ theme }) => theme.spacing.xxxl};
        }

        // Giving nested list offset to compensate for shorter SubMenuItems
        .ant-menu-sub {
            margin-top: -${({ theme }) => theme.spacing.xs};
            padding-bottom: ${({ theme }) => theme.spacing.xs};
        }

        // Subtle fade/slide-in animation when mobile subnav is opened
        &.ant-menu-submenu-open {
            .ant-menu-sub {
                animation: ${mobileNavEntrance('0.33rem')} 0.5s ease forwards;
            }
        }

        // Overriding size and color of expand/collapse caret icon
        // (using existing Ant caret icon structure)
        &,
        &:hover {
            .ant-menu-submenu-arrow {
                top: calc(50% + 2px);
                right: ${({ theme }) => theme.spacing.lg.getPxValue()}px;
                width: ${({ theme }) => theme.spacing.md.getPxValue()}px;

                &:before,
                &:after {
                    width: ${({ theme }) => theme.spacing.sm.getPxValue()}px;
                    height: 1.5px;
                    // This is how Ant handles caret icon color
                    background-image: linear-gradient(
                        90deg,
                        ${({ theme }) => `${theme.color.N7},${theme.color.N7}`}
                    );
                }

                &:before {
                    transform: rotate(-45deg)
                        translateX(${({ theme }) => theme.spacing.xxs.getPxValue() + 1.5}px);
                }

                &:after {
                    transform: rotate(45deg)
                        translateX(-${({ theme }) => theme.spacing.xxs.getPxValue() + 1.5}px);
                }
            }
        }

        // Override styles for when SubMenu is open
        &.ant-menu-submenu-open {
            // Give section title a selected color
            ${MobileSubMenuTitle},
            ${MobileLandingLink} {
                color: ${({ theme }) => theme.color.B6};
            }

            // Invert caret icon direction and change color
            .ant-menu-submenu-arrow {
                &:before,
                &:after {
                    // This is how Ant handles caret icon color
                    background-image: linear-gradient(
                        90deg,
                        ${({ theme }) => `${theme.color.B6},${theme.color.B6}`}
                    );
                }

                &:before {
                    transform: rotate(45deg)
                        translateX(${({ theme }) => theme.spacing.xxs.getPxValue() + 1.5}px);
                }

                &:after {
                    transform: rotate(-45deg)
                        translateX(-${({ theme }) => theme.spacing.xxs.getPxValue() + 1.5}px);
                }
            }
        }

        // Disable mobile interior link hover state
        ${MobileInteriorLink} {
            &:hover {
                color: ${({ theme }) => theme.color.N9};
            }
        }

        // Add a selected treatment to tapped subnav item
        .ant-menu-item-selected {
            ${MobileInteriorLink} {
                color: ${({ theme }) => theme.color.B6};
            }
        }

        .ant-menu-submenu-arrow {
            top: 34px !important;
        }
    }

    // Disable hard-coded Ant menu background and pipe styling
    &&& {
        .ant-menu-submenu-title {
            ${antSelectedItemOverrideStyles}
        }
    }
`;

// Mobile Submenu Item
const MobileSubMenuItem = styled(Menu.Item)`
    ${MobileInteriorLink} {
        padding: ${({ theme }) => `${theme.spacing.xs} ${theme.spacing.lg}`};
    }
`;
