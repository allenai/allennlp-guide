import React from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import { Menu, Icon } from 'antd';
import { Link } from '../components/Link';
import { outline } from '../outline';
import { getIcon } from '../utils';

export const Navigation = ({
    className,
    isMobile = false,
    headerLinks,
    groupedChapters,
    defaultSelectedKeys,
    defaultOpenKeys,
    onTitleClick = () => {}
}) => {
    let MenuContainer = Menu;
    let MenuItem = Menu.Item;
    let SubMenu = Menu.SubMenu;
    if (isMobile) {
        MenuContainer = MobileMenu;
        MenuItem = MobileMenuItem;
        SubMenu = MobileSubMenu;
    }

    const getMenuIcon = obj =>
        obj.antMenuIcon ? (
            <Icon type={obj.antMenuIcon} />
        ) : (
            obj.icon && <CustomIcon component={() => getIcon(obj.icon, 17)} />
        );

    return (
        <React.Fragment>
            {!isMobile && <GlobalDesktopStyle />}
            <MenuContainer
                className={className}
                defaultSelectedKeys={defaultSelectedKeys}
                defaultOpenKeys={defaultOpenKeys}
                mode="inline">
                {isMobile && (
                    <MenuItem key="home">
                        <a href="/">
                            <span>Home</span>
                        </a>
                    </MenuItem>
                )}
                <MenuItem key={outline.overview.slug}>
                    <Link to={outline.overview.slug}>
                        {!isMobile && getMenuIcon(outline.overview)}
                        <span>{groupedChapters[outline.overview.slug].node.frontmatter.title}</span>
                    </Link>
                </MenuItem>
                {outline.parts.map(
                    part =>
                        part.chapterSlugs && (
                            <SubMenu
                                key={part.title}
                                onTitleClick={() => onTitleClick && onTitleClick(part.title)}
                                title={
                                    <span>
                                        {!isMobile && getMenuIcon(part)}
                                        <span>{part.title}</span>
                                    </span>
                                }>
                                {part.chapterSlugs.map(chapterSlug => (
                                    <MenuItem key={chapterSlug}>
                                        <Link to={chapterSlug}>
                                            {groupedChapters[chapterSlug].node.frontmatter.title}
                                        </Link>
                                    </MenuItem>
                                ))}
                            </SubMenu>
                        )
                )}
                {isMobile &&
                    headerLinks.map(headerLink => (
                        <MenuItem key={headerLink.url}>
                            <Link to={headerLink.url}>
                                <span>{headerLink.text}</span>
                            </Link>
                        </MenuItem>
                    ))}
            </MenuContainer>
        </React.Fragment>
    );
};

const MobileMenu = styled(Menu)``;
const MobileMenuItem = styled(Menu.Item)``;
const MobileSubMenu = styled(Menu.SubMenu)``;

const CustomIcon = styled(Icon)``;

// Resetting Ant Menu Styles
const GlobalDesktopStyle = createGlobalStyle`
    &&& {
        .ant-menu {
            border: none !important;

            svg {
                color: ${({ theme }) => theme.color.N8};
            }

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
                    span,
                    i,
                    svg {
                        color: ${({ theme }) => theme.color.B5} !important;
                        stroke: ${({ theme }) => theme.color.B5};
                    }
                }

                .ant-menu-submenu-title:hover {
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

            // Support multi-line items without truncation
            .ant-menu-submenu-title,
            .ant-menu-item {
                overflow: visible !important;
                white-space: normal !important;
                height: auto !important;
                line-height: 1.5 !important;
                padding-top: 9px !important;
                padding-bottom 10px !important;
            }

            .ant-menu-item {
                a {
                    color: ${({ theme }) => theme.color.N10};

                    &:hover {
                        &,
                        svg {
                            color: ${({ theme }) => theme.color.B5};
                            stroke: ${({ theme }) => theme.color.B5};
                        }
                        text-decoration: none;
                    }
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
    }
`;
